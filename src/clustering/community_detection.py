import pandas as pd
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


def leiden_user_communities(driver, graph_name="userGraph"):
    with driver.session() as session:
        # Remove any existing graph with the same name
        session.run(f"CALL gds.graph.drop('{graph_name}', false)")  

        # Project the graph: only user nodes and user-user relations
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
        session.run(project_query).consume()

        # Esegui Leiden sul grafo
        leiden_query = f"""
        CALL gds.leiden.stream('{graph_name}')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).user_id AS user_id, communityId
        """
        results = session.run(leiden_query)

        # Porta i risultati in un DataFrame Pandas
        df = pd.DataFrame([dict(record) for record in results])

        # Droppa il grafo per liberare memoria
        session.run(f"CALL gds.graph.drop('{graph_name}')").consume()

    # Add user count per cluster
    community_sizes = df["communityId"].value_counts().reset_index()
    community_sizes.columns = ["communityId", "community_size"]

    # Join con il DataFrame originale
    df = df.merge(community_sizes, on="communityId", how="left")

    return df

def export_cluster_size_distribution(df, output_csv_path=None):
    """
    Take the DataFrame produced by leiden_user_communities (must contain 'community_size')
    and produce a CSV with columns:
      community_size, #communities
    Returns the distribution DataFrame.
    """
    if "community_size" not in df.columns:
        raise ValueError("Il DataFrame deve contenere la colonna 'community_size'.")

    # Considera ogni community una sola volta (in caso il df contenga più righe per community)
    communities = df[["communityId", "community_size"]].drop_duplicates(subset="communityId")

    # Conta quante community per ciascuna dimensione e ottieni un DataFrame con colonne originali
    dist = communities.groupby("community_size").size().reset_index(name="#communities")

    # Ordina per community_size e salva CSV (non rinominiamo le colonne)
    dist = dist.sort_values("community_size", ascending=False)
    if output_csv_path:
        dist.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Distribuzione completa dimensione cluster salvata in {output_csv_path}")    
    return dist


def export_users_ordered_by_cluster(df, output_csv_path=None, descending=True):
    """
    Export a CSV with user_id, communityId, community_size ordered by community_size.
    - df: DataFrame returned by leiden_user_communities (must contain 'user_id', 'communityId', 'community_size')
    - output_csv_path: output file path
    - descending: True to order largest clusters first
    Returns the saved DataFrame.
    """
    required = {"user_id", "communityId", "community_size"}
    if not required.issubset(df.columns):
        raise ValueError(f"Il DataFrame deve contenere le colonne: {required}")

    # Assicuriamoci di avere una riga per ogni user_id con il relativo cluster e dimensione
    users = df[["user_id", "communityId", "community_size"]].drop_duplicates(subset=["user_id"])

    # Ordina per dimensione del cluster (desc per default) e per communityId per stabilità
    users = users.sort_values(by=["community_size", "communityId"], ascending=[not descending, True])

    # Salva CSV con intestazione: user_id, communityId, community_size
    if output_csv_path:
        users.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Elenco completo utenti per cluster salvato in {output_csv_path}")
    return users


def analyze_communities(driver, df, max_communities=5, output_csv_path=None):
    """
    Analyzes the communities computed by leiden_user_communities.
    For each community:
      - num_users
      - top most retweeted users
      - dominant tweet class + percentage distribution
      - most frequent keywords
    If output_csv_path is specified, saves the results to a CSV ordered by community size.
    """
    community_data = []

    with driver.session() as session:     
        community_groups = df.groupby("communityId")
        # ordina per dimensione decrescente
        top_communities = community_groups.size().sort_values(ascending=False).head(max_communities).index

        for comm_id in top_communities:
            group = df[df["communityId"] == comm_id]
            if len(community_data) >= max_communities:
                break

            users = group["user_id"].tolist()

            # --- 1. Top utenti più retwittati nella community ---
            top_users_query = """
            MATCH (u:User)<-[:RETWEETED_FROM]-(v:User)
            WHERE u.user_id IN $users AND v.user_id IN $users
            RETURN u.user_id AS user, count(*) AS retweet_count
            ORDER BY retweet_count DESC
            LIMIT 3
            """
            top_users = session.run(top_users_query, users=users).data()
            top_users_str = ", ".join([f"{r['user']} ({r['retweet_count']})" for r in top_users])

            # --- 2. Tweet della community ---
            tweets_query = """
            MATCH (u:User)-[:CREATES|RETWEET]->(t:Tweet)
            WHERE u.user_id IN $users AND t.text IS NOT NULL
            RETURN t.text AS text, t.tweet_label AS label
            """
            tweets = session.run(tweets_query, users=users).data()

            texts = [t["text"] for t in tweets if t["text"]]
            labels = [t["label"] for t in tweets if t["label"]]

            # Distribuzione delle classi
            label_dist = Counter(labels)
            total = sum(label_dist.values())
            if total > 0:
                label_percentages = {k: f"{(v/total*100):.1f}%" for k, v in label_dist.items()}
                label_distribution = ", ".join([f"{k}: {p}" for k, p in label_percentages.items()])
                dominant_label = max(label_dist.keys(), key=lambda k: label_dist[k])
            else:
                label_distribution = None
                dominant_label = None

            # --- 3. Keywords ---
            # Estrai keyword dai tweet della community
            keywords = extract_keywords(texts, top_k=10)


            community_data.append({
                "community_id": comm_id,
                "num_users": len(users),
                "top_users": top_users_str,
                "dominant_label": dominant_label,
                "label_distribution": label_distribution,
                "keywords": keywords
            })

    # Mettiamo tutto in DataFrame e ordiniamo per dimensione
    df_analysis = pd.DataFrame(community_data)
    df_analysis = df_analysis.sort_values("num_users", ascending=False).reset_index(drop=True)

    if output_csv_path:
        df_analysis.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Analisi community salvata in {output_csv_path}")

    return df_analysis


# Carica modello spaCy (scegli en_core_web_sm o it_core_news_sm)
nlp = spacy.load("en_core_web_sm")  

def preprocess_texts(texts):
    """Pulisce e lemmatizza i testi con spaCy"""
    docs = nlp.pipe(texts, disable=["ner", "parser"])
    cleaned = []
    for doc in docs:
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        cleaned.append(" ".join(tokens))
    return cleaned

def extract_keywords(texts, top_k=10):
    """Estrae keywords con TF-IDF"""
    if not texts:
        return []
    cleaned = preprocess_texts(texts)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(cleaned)
    scores = X.sum(axis=0).A1 # type: ignore
    terms = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[::-1][:top_k]
    return [terms[i] for i in top_indices if terms[i]!="url"]
