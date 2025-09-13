import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import numpy as np
from src.utils.neo4j_utils import get_neo4j_driver

# =========================================
# 2. Connessione a Neo4j
# =========================================
driver = get_neo4j_driver()

# =========================================
# 3. Creazione grafo utenti per embeddings
# =========================================
def create_user_graph(driver, graph_name="userGraph"):
    with driver.session() as session:
        # Droppa il grafo se esiste già
        try:
            session.run(f"CALL gds.graph.drop('userGraph') YIELD graphName;")
            print(f"Grafo '{graph_name}' eliminato.")
        except Exception:
            print(f"Nessun grafo '{graph_name}' da eliminare.")

        # Crea il nuovo grafo proiettando solo utenti e relazioni RETWEETED_FROM
        project_query = f"""
        CALL gds.graph.project(
          '{graph_name}',
          'User',
          {{
            RETWEETED_FROM: {{orientation: 'UNDIRECTED'}}
          }}
        )
        YIELD graphName, nodeCount, relationshipCount
        """
        result = session.run(project_query).data()[0]
        print(f"Grafo '{result['graphName']}' creato con {result['nodeCount']} nodi e {result['relationshipCount']} archi.")


# =========================================
# 4. Generazione embeddings utenti con FastRP
# =========================================
def generate_user_embeddings(driver, embedding_dim=64, graph_name="userGraph"):
    with driver.session() as session:
        # Step 1: genera embedding con FastRP e li salva come proprietà
        mutate_query = f"""
        CALL gds.fastRP.mutate(
          '{graph_name}',
          {{
            embeddingDimension: $embedding_dim,
            mutateProperty: 'embedding'
          }}
        )
        YIELD nodePropertiesWritten
        """
        session.run(mutate_query, embedding_dim=embedding_dim).consume()

        # Step 2: recupera gli embedding
        stream_query = f"""
        CALL gds.graph.nodeProperty.stream('{graph_name}', 'embedding')
        YIELD nodeId, propertyValue
        RETURN gds.util.asNode(nodeId).user_id AS user_id, propertyValue AS embedding
        """
        results = session.run(stream_query)

        # Step 3: converti in DataFrame
        df = pd.DataFrame([dict(record) for record in results])
        return df


create_user_graph(driver)
user_embeddings = generate_user_embeddings(driver, embedding_dim=64)
# If generate_user_embeddings returned a DataFrame, convert to dict user_id -> numpy array
if isinstance(user_embeddings, pd.DataFrame):
    emb_dict = {}
    for _, row in user_embeddings.iterrows():
        uid = str(row.get("user_id"))
        emb = row.get("embedding") or row.get("propertyValue") or row.get("property_value")
        # ensure embedding is numpy array
        emb = np.array(emb, dtype=float)
        emb_dict[uid] = emb
    user_embeddings = emb_dict

print(f"Embeddings utenti generati: {len(user_embeddings)} utenti")

# =========================================
# 5. Caricamento dataset (user, tweet, delay)
# =========================================
def load_training_data(driver, limit=10000):
    query = f"""
    MATCH (u:User)-[r:RETWEET]->(t:Tweet)
    WHERE t.text IS NOT NULL AND r.delay IS NOT NULL
    RETURN u.user_id AS user_id, t.text AS text, r.delay AS delay
    LIMIT {limit}
    """
    with driver.session() as session:
        results = session.run(query)
        return pd.DataFrame([dict(record) for record in results])

data = load_training_data(driver)
print(f"Dataset caricato: {len(data)} esempi")

# --- NEW: ensure user ids are strings and filter rows lacking embeddings ---
data['user_id'] = data['user_id'].astype(str)
valid_mask = data['user_id'].isin(set(user_embeddings.keys()))
if not valid_mask.any():
    raise RuntimeError("No training rows have corresponding user embeddings. Check embeddings generation and user IDs.")
# Filter data to only rows with embeddings
data = data[valid_mask].reset_index(drop=True)
print(f"Dataset filtered to {len(data)} examples with embeddings available")

# =========================================
# 6. Preprocessing testi (TF-IDF + SVD)
# =========================================
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = vectorizer.fit_transform(data["text"])

svd = TruncatedSVD(n_components=128)
X_text = svd.fit_transform(X_tfidf)

# =========================================
# 7. Costruzione dataset (features + labels)
# =========================================
# Build X_user aligned with filtered data
X_user = np.vstack([user_embeddings[str(uid)] for uid in data["user_id"]])
X = np.hstack([X_user, X_text])

y_prob = np.ones(len(data))  # tutti esempi = retweet avvenuto (1)
y_delay = data["delay"].values

# split train/val
X_train, X_val, y_prob_train, y_prob_val, y_delay_train, y_delay_val = train_test_split(
    X, y_prob, y_delay, test_size=0.2, random_state=42
)

# conversione a tensori
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_prob_train_t = torch.tensor(y_prob_train, dtype=torch.float32).unsqueeze(1)
y_prob_val_t = torch.tensor(y_prob_val, dtype=torch.float32).unsqueeze(1)
y_delay_train_t = torch.tensor(y_delay_train, dtype=torch.float32).unsqueeze(1)
y_delay_val_t = torch.tensor(y_delay_val, dtype=torch.float32).unsqueeze(1)

# =========================================
# 8. Definizione modello
# =========================================
class PropagationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.out_prob = nn.Linear(128, 1)
        self.out_delay = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        prob = self.sigmoid(self.out_prob(h))
        delay = self.out_delay(h)
        return prob, delay

in_dim = X.shape[1]
model = PropagationModel(in_dim)

# =========================================
# 9. Training con validazione
# =========================================
criterion_prob = nn.BCELoss()
criterion_delay = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    # train
    model.train()
    optimizer.zero_grad()
    prob_pred, delay_pred = model(X_train_t)
    loss_prob = criterion_prob(prob_pred, y_prob_train_t)
    loss_delay = criterion_delay(delay_pred, y_delay_train_t)
    loss = loss_prob + loss_delay
    loss.backward()
    optimizer.step()

    # validazione
    model.eval()
    with torch.no_grad():
        prob_val, delay_val = model(X_val_t)
        val_loss_prob = criterion_prob(prob_val, y_prob_val_t)
        val_loss_delay = criterion_delay(delay_val, y_delay_val_t)
        val_loss = val_loss_prob + val_loss_delay

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {loss.item():.4f} - "
          f"Val Loss: {val_loss.item():.4f}")

# =========================================
# 10. Valutazione finale
# =========================================
model.eval()
with torch.no_grad():
    prob_val, delay_val = model(X_val_t)
    prob_val_np = prob_val.numpy()
    delay_val_np = delay_val.numpy()

# accuracy per probabilità (threshold=0.5)
acc = accuracy_score(y_prob_val, (prob_val_np > 0.5).astype(int))
# MSE e R² per delay
mse = mean_squared_error(y_delay_val, delay_val_np)
r2 = r2_score(y_delay_val, delay_val_np)

print("\n=== Valutazione finale ===")
print(f"Accuracy previsione retweet: {acc:.3f}")
print(f"MSE delay: {mse:.3f}")
print(f"R² delay: {r2:.3f}")

# =========================================
# 11. Funzione di predizione globale
# =========================================
def predict_global_propagation(model, user_embeddings, tweet_text, vectorizer, svd):
    t_vec = vectorizer.transform([tweet_text])
    t_vec = svd.transform(t_vec)[0]

    probs, delays, users = [], [], []
    with torch.no_grad():
        for u_id, u_emb in user_embeddings.items():
            x = np.hstack([u_emb, t_vec])
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            prob, delay = model(x)
            probs.append(prob.item())
            delays.append(delay.item())
            users.append(u_id)

    total_expected_retweets = sum(probs)
    weighted_delay = sum(p * d for p, d in zip(probs, delays)) / (total_expected_retweets + 1e-6)

    top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    top_users = [(users[i], probs[i], delays[i]) for i in top_idx]

    return {
        "total_expected_retweets": total_expected_retweets,
        "avg_delay": weighted_delay,
        "top_users": top_users
    }

# =========================================
# 12. Esempio di utilizzo
# =========================================
new_tweet = "Breaking news: AI is transforming social media analytics!"
result = predict_global_propagation(model, user_embeddings, new_tweet, vectorizer, svd)

print("\n=== Predizione globale ===")
print(f"Numero atteso di retweet: {result['total_expected_retweets']:.1f}")
print(f"Delay medio previsto: {result['avg_delay']:.2f} secondi")
print("Utenti più probabili:")
for u, p, d in result["top_users"]:
    print(f"  - User {u}: prob={p:.2f}, delay={d:.1f}s")
