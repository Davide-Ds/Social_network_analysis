import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import numpy as np


# =========================================
# 3. Create user graph for embeddings
# =========================================
def user_graph(driver, drop = False, graph_name="userGraph"):
    with driver.session() as session:
        if drop:
            # Drop the graph if it already exists
            try:
                session.run(f"CALL gds.graph.drop('userGraph') YIELD graphName;")
                print(f"Graph '{graph_name}' dropped.")
            except Exception:
                print(f"No graph '{graph_name}' to drop.")
        else:
            # Create the new projected graph with users and RETWEETED_FROM relationships
            project_query = f"""
            CALL gds.graph.project(
            '{graph_name}',
            'User',
            {{
                RETWEETED_FROM: {{orientation: 'NATURAL'}}
            }}
            )
            YIELD graphName, nodeCount, relationshipCount
            """
            result = session.run(project_query).data()[0]
            print(f"Graph '{result['graphName']}' created with {result['nodeCount']} nodes and {result['relationshipCount']} relationships.")


# =========================================
# 4. Generate user embeddings with FastRP
# =========================================
def generate_user_embeddings(driver, embedding_dim=64, graph_name="userGraph"):
    with driver.session() as session:
        # Step 1: generate embeddings with FastRP and store as a property
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

        # Step 2: retrieve embeddings
        stream_query = f"""
        CALL gds.graph.nodeProperty.stream('{graph_name}', 'embedding')
        YIELD nodeId, propertyValue
        RETURN gds.util.asNode(nodeId).user_id AS user_id, propertyValue AS embedding
        """
        results = session.run(stream_query)

        # Step 3: convert to DataFrame
        df = pd.DataFrame([dict(record) for record in results])
        
        # ensure embedding is numpy array
        emb_dict = {}
        for _, row in df.iterrows():
            uid = str(row.get("user_id"))
            emb = row.get("embedding") or row.get("propertyValue") or row.get("property_value")
            emb = np.array(emb, dtype=float)
            emb_dict[uid] = emb
        return emb_dict


# =========================================
# 5. Load dataset (user, tweet, delay)
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


# =========================================
# 6. Text preprocessing (TF-IDF + SVD)
# =========================================
def build_features_and_dataset(data, user_embeddings, tfidf_max_features=5000, svd_components=128):
    # ensure user ids are strings and filter rows lacking embeddings
    data = data.copy()
    data['user_id'] = data['user_id'].astype(str)
    valid_mask = data['user_id'].isin(set(user_embeddings.keys()))
    if not valid_mask.any():
        raise RuntimeError("No training rows have corresponding user embeddings. Check embeddings generation and user IDs.")
    data = data[valid_mask].reset_index(drop=True)

    # text preprocessing: TF-IDF + SVD
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
    X_tfidf = vectorizer.fit_transform(data["text"])
    svd = TruncatedSVD(n_components=svd_components)
    X_text = svd.fit_transform(X_tfidf)

    # user embeddings aligned with filtered data
    X_user = np.vstack([user_embeddings[str(uid)] for uid in data["user_id"]])
    X = np.hstack([X_user, X_text])

    y_prob = np.ones(len(data))  # observed retweets
    y_delay = data["delay"].values

    # split
    X_train, X_val, y_prob_train, y_prob_val, y_delay_train, y_delay_val = train_test_split(
        X, y_prob, y_delay, test_size=0.2, random_state=42
    )

    tensors = {
        "X_train_t": torch.tensor(X_train, dtype=torch.float32),
        "X_val_t": torch.tensor(X_val, dtype=torch.float32),
        "y_prob_train_t": torch.tensor(y_prob_train, dtype=torch.float32).unsqueeze(1),
        "y_prob_val_t": torch.tensor(y_prob_val, dtype=torch.float32).unsqueeze(1),
        "y_delay_train_t": torch.tensor(y_delay_train, dtype=torch.float32).unsqueeze(1),
        "y_delay_val_t": torch.tensor(y_delay_val, dtype=torch.float32).unsqueeze(1),
        "vectorizer": vectorizer,
        "svd": svd,
        "filtered_data": data
    }
    return tensors


# =========================================
# 8. Model definition
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


# =========================================
# 9. Training and validation
# =========================================
def train_model(model, tensors, epochs=10, lr=1e-3):
    criterion_prob = nn.BCELoss()
    criterion_delay = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = tensors["X_train_t"]
    y_prob_train_t = tensors["y_prob_train_t"]
    y_delay_train_t = tensors["y_delay_train_t"]
    X_val_t = tensors["X_val_t"]
    y_prob_val_t = tensors["y_prob_val_t"]
    y_delay_val_t = tensors["y_delay_val_t"]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prob_pred, delay_pred = model(X_train_t)
        loss_prob = criterion_prob(prob_pred, y_prob_train_t)
        loss_delay = criterion_delay(delay_pred, y_delay_train_t)
        loss = loss_prob + loss_delay
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prob_val, delay_val = model(X_val_t)
            val_loss_prob = criterion_prob(prob_val, y_prob_val_t)
            val_loss_delay = criterion_delay(delay_val, y_delay_val_t)
            val_loss = val_loss_prob + val_loss_delay

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")
    return model


# =========================================
# 10. Final evaluation
# =========================================
def evaluate_model(model, tensors):
    model.eval()
    with torch.no_grad():
        prob_val, delay_val = model(tensors["X_val_t"])
        prob_val_np = prob_val.numpy()
        delay_val_np = delay_val.numpy()

    acc = accuracy_score(tensors["y_prob_val_t"].numpy(), (prob_val_np > 0.5).astype(int))
    mse = mean_squared_error(tensors["y_delay_val_t"].numpy(), delay_val_np)
    r2 = r2_score(tensors["y_delay_val_t"].numpy(), delay_val_np)
    return {"accuracy": float(acc), "mse": float(mse), "r2": float(r2)}


# =========================================
# 11. Global prediction function
# =========================================
def predict_global_propagation(model, user_embeddings, tweet_text, vectorizer, svd, top_k=5):
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

    top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
    top_users = [(users[i], probs[i], delays[i]) for i in top_idx]

    return {
        "total_expected_retweets": total_expected_retweets,
        "avg_delay": weighted_delay,
        "top_users": top_users
    }


# =========================================
# 12. Usage example
# =========================================
def tweet_propagation_prediction_NN(driver, tweet_text, limit=10000, embedding_dim=64, svd_components=128, tfidf_max_features=5000, epochs=10):

    # graph + embeddings
    user_graph(driver)
    user_embeddings = generate_user_embeddings(driver, embedding_dim=embedding_dim)
    print(f"Generated user embeddings: {len(user_embeddings)} users")

    # load & prepare data
    data = load_training_data(driver, limit=limit)
    print(f"Loaded dataset: {len(data)} examples")
    tensors = build_features_and_dataset(data, user_embeddings, tfidf_max_features=tfidf_max_features, svd_components=svd_components)

    # create, train and evaluate model
    input_dim = tensors["X_train_t"].shape[1]
    model = PropagationModel(input_dim)
    model = train_model(model, tensors, epochs=epochs)
    metrics = evaluate_model(model, tensors)
    print("\n=== Final evaluation ===")
    print(metrics)

    # example prediction
    res = predict_global_propagation(model, user_embeddings, tweet_text, tensors["vectorizer"], tensors["svd"])
    print("\n=== Global prediction ===")
    print(f"Expected number of retweets: {res['total_expected_retweets']:.1f}")
    print(f"Predicted average delay: {res['avg_delay']:.2f} seconds")
    print("Top users likely to retweet:")
    for u, p, d in res["top_users"]:
        print(f" - User {u}: prob={p:.2f}, delay={d:.1f}s")
    
    user_graph(driver, drop=True)
    driver.close()