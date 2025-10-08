import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# =========================================
# 1. Create user graph for embeddings
# =========================================
def user_graph(driver, drop = False, graph_name="userGraph"):
    """
    Create or drop a GDS projected graph containing User nodes and RETWEETED_FROM relationships.

    Args:
        driver: Neo4j driver instance (authenticated session manager).
        drop (bool): If True, attempt to drop an existing GDS graph with name graph_name.
        graph_name (str): Name used to project the GDS graph.

    Behavior:
        - If drop is True, tries to drop an existing graph and prints the result.
        - Otherwise projects a new graph using gds.graph.project over User nodes and RETWEETED_FROM relationships.

    Notes:
        - This function uses the Neo4j GDS plugin; ensure GDS is installed and available in the DB.
        - The function prints summary info (graphName, nodeCount, relationshipCount).
    """
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
# 2. Generate user embeddings with FastRP
# =========================================
def generate_user_embeddings(driver, embedding_dim=64, graph_name="userGraph"):
    """
    Generate node embeddings using GDS FastRP and return a dict user_id -> numpy array embedding.

    Args:
        driver: Neo4j driver instance.
        embedding_dim (int): Dimension of the FastRP embedding to compute.
        graph_name (str): GDS graph name where FastRP will be executed.

    Returns:
        dict: Mapping from user_id (string) to L2-normalized numpy embedding vector.

    Behavior:
        - Calls gds.fastRP.mutate to write an 'embedding' node property.
        - Streams node property values back and converts to numpy arrays.
        - L2-normalizes each vector to improve numeric stability when using cosine-like comparisons.

    Notes:
        - If embedding vectors are stored in alternative keys, the function attempts multiple property names.
        - Returned embeddings may be used directly for concatenation with text features.
    """
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
        
        # ensure embedding is numpy array and L2-normalize
        emb_dict = {}
        for _, row in df.iterrows():
            uid = str(row.get("user_id"))
            emb = row.get("embedding") or row.get("propertyValue") or row.get("property_value")
            emb = np.array(emb, dtype=float)
            # L2 normalize embedding vector
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            emb_dict[uid] = emb
        return emb_dict


# =========================================
# 3. Load dataset (user, tweet, delay)
# =========================================
def load_training_data(driver, limit=200000):
    """
    Load training examples (user, tweet text, delay) from Neo4j.

    Args:
        driver: Neo4j driver instance.
        limit (int): Maximum number of rows to fetch.

    Returns:
        pandas.DataFrame with columns ['user_id', 'text', 'delay'] where delay is numeric minutes.

    Behavior:
        - Executes a Cypher query matching (u)-[r:RETWEET]->(t) and returns user id, tweet text and retweet delay.
        - Attempts to coerce delay column to numeric and returns the DataFrame.
    """
    query = f"""
    MATCH (u:User)-[r:RETWEET]->(t:Tweet)
    WHERE t.text IS NOT NULL AND r.delay IS NOT NULL
    RETURN u.user_id AS user_id, t.text AS text, r.delay AS delay
    LIMIT {limit}
    """
    with driver.session() as session:
        results = session.run(query)
        df = pd.DataFrame([dict(record) for record in results])
    # convert delay to float (minutes as specified)
    if 'delay' in df.columns:
        df['delay'] = pd.to_numeric(df['delay'], errors='coerce')
    return df


# =========================================
# 4. Text preprocessing (TF-IDF + SVD) and feature build
# =========================================
def build_features_and_dataset(data, user_embeddings, tfidf_max_features=5000, svd_components=128, negative_ratio=0.0, random_state=42):
    """
    Construct features and prepare datasets for the classifier and regressor.

    Args:
        data (pandas.DataFrame): DataFrame with columns ['user_id','text','delay'] (positives only).
        user_embeddings (dict): Mapping user_id -> numpy embedding vector.
        tfidf_max_features (int): max features for the TF-IDF vectorizer.
        svd_components (int): number of SVD components to retain.
        negative_ratio (float): ratio of negative samples to positive samples for classifier.
        random_state (int): RNG seed for reproducibility.

    Returns:
        dict containing torch tensors for training/validation sets and fitted vectorizer/svd/scaler.
        Keys include X_train_t, X_val_t, y_delay_train_t, y_delay_val_t and optionally
        X_clf_train_t, X_clf_val_t, y_clf_train_t, y_clf_val_t when negative_ratio>0.

    Processing steps:
        - Filters rows missing embeddings, text, or delay.
        - Computes log1p(delay) to stabilize heavy-tailed distribution.
        - Builds TF-IDF matrix from tweet text and reduces dimensionality with TruncatedSVD.
        - Standardizes text components, normalizes user embeddings, concatenates features.
        - If negative_ratio>0, performs negative sampling to create classifier dataset.
        - Splits data into train/val for regressor and classifier (if present).

    Notes:
        - The classifier negatives are sampled from users that did not retweet the corresponding tweet.
        - Returned tensors are ready to be wrapped into DataLoader for minibatch training.
    """
    # ensure user ids are strings and filter rows lacking embeddings
    data = data.copy()
    data['user_id'] = data['user_id'].astype(str)
    valid_mask = data['user_id'].isin(set(user_embeddings.keys())) & data['delay'].notnull() & data['text'].notnull()
    if not valid_mask.any():
        raise RuntimeError("No training rows have corresponding user embeddings or valid text/delay.")
    data = data[valid_mask].reset_index(drop=True)

    # delay is in minutes -> apply log1p transform to stabilize variance
    delays = data['delay'].astype(float).values
    print("Delay stats (minutes):", "count=", len(delays), "min=", float(np.nanmin(delays)), "median=", float(np.nanmedian(delays)), "mean=", float(np.nanmean(delays)), "max=", float(np.nanmax(delays)))
    y_delay = np.log1p(delays)

    # text preprocessing: TF-IDF + SVD
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
    X_tfidf = vectorizer.fit_transform(data["text"])
    svd = TruncatedSVD(n_components=svd_components)
    X_text = svd.fit_transform(X_tfidf)
    # standardize text features
    scaler_text = StandardScaler()
    X_text = scaler_text.fit_transform(X_text)

    # user embeddings aligned with filtered data and ensure shape
    X_user = np.vstack([user_embeddings[str(uid)] for uid in data["user_id"]])
    # if embeddings have different dim than expected, pad or trim (robustness)
    # combine and scale
    X_user = normalize(X_user, axis=1)

    # combine features
    X = np.hstack([X_user, X_text])

    # classification dataset with negative sampling
    clf_tensors = {}
    if negative_ratio and negative_ratio > 0.0:
        rng = np.random.RandomState(random_state)
        pos_count = X.shape[0]
        neg_count = int(pos_count * negative_ratio)
        user_pool = list(user_embeddings.keys())
        # exclude positive users to avoid accidental positives
        positive_users = set(data['user_id'].tolist())
        candidates = [u for u in user_pool if u not in positive_users]
        if len(candidates) == 0:
            raise RuntimeError("No candidate users for negative sampling. Reduce negative_ratio or provide more embeddings.")
        neg_users = rng.choice(candidates, size=neg_count, replace=len(candidates) < neg_count)
        # sample texts indices to pair with neg users
        neg_text_idx = rng.randint(0, X_text.shape[0], size=neg_count)
        X_user_neg = np.vstack([user_embeddings[str(u)] for u in neg_users])
        X_text_neg = X_text[neg_text_idx]
        X_neg = np.hstack([normalize(X_user_neg, axis=1), X_text_neg])

        X_clf = np.vstack([X, X_neg])
        y_clf = np.concatenate([np.ones(X.shape[0], dtype=np.float32), np.zeros(X_neg.shape[0], dtype=np.float32)])

        X_clf_train, X_clf_val, y_clf_train, y_clf_val = train_test_split(X_clf, y_clf, test_size=0.2, random_state=random_state, stratify=y_clf)

        clf_tensors = {
            'X_clf_train_t': torch.tensor(X_clf_train, dtype=torch.float32),
            'X_clf_val_t': torch.tensor(X_clf_val, dtype=torch.float32),
            'y_clf_train_t': torch.tensor(y_clf_train, dtype=torch.float32).unsqueeze(1),
            'y_clf_val_t': torch.tensor(y_clf_val, dtype=torch.float32).unsqueeze(1)
        }

    # split for regression (only positives)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_delay, test_size=0.2, random_state=random_state
    )

    tensors = {
        "X_train_t": torch.tensor(X_train, dtype=torch.float32),
        "X_val_t": torch.tensor(X_val, dtype=torch.float32),
        "y_delay_train_t": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        "y_delay_val_t": torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        "vectorizer": vectorizer,
        "svd": svd,
        "scaler_text": scaler_text,
        "filtered_data": data
    }
    tensors.update(clf_tensors)
    return tensors


# =========================================
# 5. Model definition (regression on log1p(delay))
# =========================================
class PropagationModel(nn.Module):
    """
    Regression model predicting log1p(delay) for a user-tweet pair.

    Architecture:
        - Fully connected: input_dim -> hidden_dim -> 128 -> 1
        - ReLU activations

    Usage:
        - Input: torch tensor of shape (batch_size, input_dim) containing concatenated user and text features.
        - Output: single continuous value per example representing predicted log1p(delay).
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.out_delay = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        delay = self.out_delay(h)
        return delay


# Classifier model
class ClassifierModel(nn.Module):
    """
    Binary classifier for retweet probability of a user on a given tweet.

    Architecture:
        - Fully connected network with two hidden layers and a final sigmoid output.

    Usage:
        - Input: concatenated features (user embedding + tweet text embedding).
        - Output: probability in [0,1] (retweet likelihood).
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.out = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.sigmoid(self.out(h))


# =========================================
# 6. Training and validation (MSE on log1p delay)
# =========================================
def train_model(model, tensors, epochs=10, lr=1e-3, batch_size=256, patience=5):
    """
    Train the regression model using minibatches and SmoothL1 (Huber) loss.

    Args:
        model (nn.Module): regression model instance (PropagationModel).
        tensors (dict): dataset tensors as returned by build_features_and_dataset.
        epochs (int): maximum number of epochs.
        lr (float): learning rate for optimizer.
        batch_size (int): minibatch size.
        patience (int): early stopping patience based on validation loss.

    Returns:
        Trained model (best validation weights restored if early stopped).

    Notes:
        - Uses Adam optimizer and SmoothL1Loss for robustness to outliers.
        - Prints per-epoch train/validation loss prefixed with [REGRESSOR].
    """
    # Use SmoothL1 (Huber) for robustness to outliers and minibatch training
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(tensors["X_train_t"], tensors["y_delay_train_t"])
    val_ds = TensorDataset(tensors["X_val_t"], tensors["y_delay_val_t"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"[REGRESSOR] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # early stopping on val_loss
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[REGRESSOR] Early stopping at epoch {epoch+1}")
                break

    # load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


# train classifier
def train_classifier(model, tensors, epochs=10, lr=1e-3, batch_size=256, patience=5):
    """
    Train the binary classifier using minibatches and BCE loss.

    Args:
        model (nn.Module): classifier instance (ClassifierModel).
        tensors (dict): dataset tensors including X_clf_train_t, y_clf_train_t, etc.
        epochs, lr, batch_size, patience: training hyperparameters.

    Returns:
        Trained classifier model (best validation state restored if early stopped).

    Prints:
        - Per-epoch training loss, validation loss and validation AUC (or 'n/a' if not computable).
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(tensors['X_clf_train_t'], tensors['y_clf_train_t'])
    val_ds = TensorDataset(tensors['X_clf_val_t'], tensors['y_clf_val_t'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float('inf')
    best_state = None
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        ys, probs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = model(xb)
                val_losses.append(float(nn.BCELoss()(p, yb)))
                ys.append(yb.numpy().ravel())
                probs.append(p.numpy().ravel())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        ys = np.concatenate(ys) if ys else np.array([])
        probs = np.concatenate(probs) if probs else np.array([])
        auc = float(roc_auc_score(ys, probs)) if ys.size and len(np.unique(ys))>1 else None
        auc_str = f"{auc:.4f}" if auc is not None else "n/a"
        print(f"[CLASSIFIER] Epoch {epoch+1}/{epochs} - Train Loss: {np.mean(train_losses):.4f} - Val Loss: {val_loss:.4f} - Val AUC: {auc_str}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[CLASSIFIER] Early stopping at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =========================================
# 7. Final evaluation
# =========================================
def evaluate_model(model, tensors, batch_size=512):
    """
    Evaluate the regression model on the validation set and return metrics on the original minutes scale.

    Args:
        model (nn.Module): trained regression model that outputs log1p(delay).
        tensors (dict): dataset tensors produced by build_features_and_dataset.
        batch_size (int): evaluation batch size.

    Returns:
        dict with keys: mse, r2, mae, medae — all computed on original delay scale (minutes).

    Notes:
        - Inverse transform is applied using expm1 to map predictions back to minutes.
        - MSE is sensitive to large outliers; median absolute error (medae) is more robust.
    """
    model.eval()
    val_ds = TensorDataset(tensors["X_val_t"], tensors["y_delay_val_t"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb).numpy().ravel()
            preds.append(out)
            trues.append(yb.numpy().ravel())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    # inverse log1p
    y_true = np.expm1(trues)
    y_pred = np.expm1(preds)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    medae = np.median(np.abs(y_true - y_pred))
    return {"mse": float(mse), "r2": float(r2), "mae": float(mae), "medae": float(medae)}


# =========================================
# 8. Global prediction function (predict delay in minutes)
# =========================================
def predict_global_propagation(model_reg, user_embeddings, tweet_text, vectorizer, svd, scaler_text, model_clf=None, top_k=5):
    """
    Predict top-k users likely to retweet soon for a given tweet.

    Args:
        model_reg: trained regression model (predicts log1p delay).
        user_embeddings: dict user_id->embedding.
        tweet_text: raw tweet text to vectorize.
        vectorizer/svd/scaler_text: fitted text preprocessing objects.
        model_clf (optional): trained classifier to pre-select high-probability users.
        top_k (int): number of users to return.

    Returns:
        dict containing "top_users": list of tuples (user_id, prob_or_None, predicted_delay_minutes).

    Behavior:
        - If classifier provided, compute probabilities over all users, take top candidates and rank by prob then delay.
        - Otherwise use regressor over all users and return smallest predicted delays.
    """
    t_vec = vectorizer.transform([tweet_text])
    t_vec = svd.transform(t_vec)[0]
    t_vec = scaler_text.transform(t_vec.reshape(1, -1))[0]

    results = []
    with torch.no_grad():
        if model_clf is not None:
            # compute probabilities for all users
            probs = []
            users = []
            for u_id, u_emb in user_embeddings.items():
                u = u_emb / (np.linalg.norm(u_emb) + 1e-12)
                x = np.hstack([u, t_vec])
                p = model_clf(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
                probs.append(p)
                users.append((u_id, u))
            # select top candidates by probability
            idx_sorted = np.argsort(probs)[::-1][:top_k*5]
            candidates = [users[i] for i in idx_sorted]
            for u_id, u_emb in candidates:
                x = np.hstack([u_emb / (np.linalg.norm(u_emb)+1e-12), t_vec])
                log_delay = model_reg(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
                delay = float(np.expm1(log_delay))
                p = probs[users.index((u_id, u_emb))]
                results.append((u_id, p, delay))
            # sort by probability then delay
            results = sorted(results, key=lambda x: (-x[1], x[2]))[:top_k]
        else:
            # fallback: predict delay for all users
            for u_id, u_emb in user_embeddings.items():
                u = u_emb / (np.linalg.norm(u_emb) + 1e-12)
                x = np.hstack([u, t_vec])
                log_delay = model_reg(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
                delay = float(np.expm1(log_delay))
                results.append((u_id, None, delay))
            results = sorted(results, key=lambda x: x[2])[:top_k]

    return {"top_users": results}


# Predict expected number of retweets for a given tweet
def predict_expected_retweets(model_reg, user_embeddings, tweet_text, vectorizer, svd, scaler_text, model_clf=None, cutoff_minutes=1440):
    """
    Estimate expected number of retweets for a tweet.

    Strategy:
        - If classifier available: sum predicted probabilities across users (expected value of retweet count).
        - Otherwise: use the regressor to predict delays and count users whose predicted delay <= cutoff_minutes.

    Returns:
        dict with either {'expected_retweets': float} or {'predicted_retweeters_count': int, 'cutoff_minutes': int}.

    Notes:
        - When summing probabilities, consider calibrating classifier probabilities for better absolute counts.
        - The cutoff-based fallback is a crude heuristic and depends on chosen cutoff_minutes.
    """
    t_vec = vectorizer.transform([tweet_text])
    t_vec = svd.transform(t_vec)[0]
    t_vec = scaler_text.transform(t_vec.reshape(1, -1))[0]

    if model_clf is not None:
        total_prob = 0.0
        with torch.no_grad():
            for u_id, u_emb in user_embeddings.items():
                u = u_emb / (np.linalg.norm(u_emb) + 1e-12)
                x = np.hstack([u, t_vec])
                p = model_clf(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
                total_prob += p
        return {"expected_retweets": float(total_prob)}
    else:
        # fallback: use regressor to predict delay and count those within cutoff
        count = 0
        with torch.no_grad():
            for u_id, u_emb in user_embeddings.items():
                u = u_emb / (np.linalg.norm(u_emb) + 1e-12)
                x = np.hstack([u, t_vec])
                log_delay = model_reg(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
                delay = float(np.expm1(log_delay))
                if delay <= cutoff_minutes:
                    count += 1
        return {"predicted_retweeters_count": int(count), "cutoff_minutes": cutoff_minutes}


# =========================================
# 9. Usage example
# =========================================
def tweet_propagation_prediction_NN(driver, tweet_text, limit=15000, embedding_dim=64, svd_components=128, tfidf_max_features=5000, epochs=10, batch_size=256, negative_ratio=1.0):
    """
    High-level pipeline wrapping the whole propagation prediction flow.

    Steps performed:
        1. Create/load user graph and generate user embeddings (FastRP).
        2. Load a dataset of (user, tweet text, delay) examples from Neo4j.
        3. Build features for classifier/regressor (TF-IDF + SVD for text, normalized user embeddings).
        4. Optionally train a binary classifier with negative sampling to predict retweet probability.
        5. Train a regression model on positive examples to predict log1p(delay).
        6. Evaluate regressor and compute expected retweet counts and top-k user predictions.

    Args:
        driver: Neo4j driver.
        tweet_text: example tweet text to run final predictions on.
        limit: maximum rows to fetch from DB.
        embedding_dim, svd_components, tfidf_max_features: feature/model hyperparameters.
        epochs, batch_size: training hyperparameters.
        negative_ratio: number of negative samples (per positive) for classifier.

    Returns:
        None; prints progress, metrics and top-k predictions. Close DB connection on exit.
    """
    # graph + embeddings
    user_graph(driver)
    user_embeddings = generate_user_embeddings(driver, embedding_dim=embedding_dim)
    print(f"Generated user embeddings: {len(user_embeddings)} users")

    # load & prepare data (include negative samples for classifier)
    tensors = build_features_and_dataset(load_training_data(driver, limit=limit), user_embeddings, tfidf_max_features=tfidf_max_features, svd_components=svd_components, negative_ratio=negative_ratio)
    data = tensors['filtered_data']
    print(f"Loaded dataset: {len(data)} positive examples")

    # baseline (median) on original scale
    delays = data['delay'].astype(float).values
    median_delay = np.median(delays)
    baseline_mse = mean_squared_error(delays, np.full_like(delays, median_delay))
    print(f"Baseline median delay: {median_delay:.2f} minutes, baseline MSE: {baseline_mse:.2f}")

    # train classifier
    if negative_ratio > 0.0:
        input_dim = tensors['X_clf_train_t'].shape[1]
        clf = ClassifierModel(input_dim)
        clf = train_classifier(clf, tensors, epochs=epochs, batch_size=batch_size)
    else:
        clf = None

    # train regressor on positives
    input_dim = tensors["X_train_t"].shape[1]
    reg = PropagationModel(input_dim)
    reg = train_model(reg, tensors, epochs=epochs, batch_size=batch_size)

    metrics = evaluate_model(reg, tensors)
    print("\n=== Final evaluation (regressor) ===")
    print(metrics)

    # compute expected retweets using classifier (if available) or regressor fallback
    expected = predict_expected_retweets(reg, user_embeddings, tweet_text, tensors["vectorizer"], tensors["svd"], tensors["scaler_text"], model_clf=clf)
    if "expected_retweets" in expected:
        print(f"Estimated expected retweets (sum of probabilities): {expected['expected_retweets']:.1f}")
    else:
        print(f"Estimated number of retweeters within {expected.get('cutoff_minutes', 1440)} minutes: {expected.get('predicted_retweeters_count')}")

    # example prediction using classifier + regressor
    res = predict_global_propagation(reg, user_embeddings, tweet_text, tensors["vectorizer"], tensors["svd"], tensors["scaler_text"], model_clf=clf, top_k=5)
    print("\n=== Global prediction ===")
    print(f"Top users predicted to retweet soon (user_id, prob, predicted_delay_minutes):")
    for u, p, d in res["top_users"]:
        print(f" - User {u}: prob={p:.3f}, predicted delay={d:.2f} minutes")
    
    user_graph(driver, drop=True)
    driver.close()