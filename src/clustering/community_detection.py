import pandas as pd
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


def leiden_user_communities(driver, graph_name="userGraph"):
    """
    Detects communities of users using the Leiden algorithm on a graph.
    
    The graph includes:    
        - User nodes
        - Relationships: RETWEETED_FROM (undirected)
    
    Args:
        driver: Neo4j driver.
        graph_name: name of the GDS graph to create/use.
    
    Returns:
        DataFrame with user_id and communityId for each user.
    """
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

        # Execute the Leiden algorithm
        leiden_query = f"""
        CALL gds.leiden.stream('{graph_name}')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).user_id AS user_id, communityId
        """
        results = session.run(leiden_query)

        # Convert results to a Pandas DataFrame
        df = pd.DataFrame([dict(record) for record in results])

        # Drop the graph to free memory
        session.run(f"CALL gds.graph.drop('{graph_name}')").consume()

    # Add user count per cluster
    community_sizes = df["communityId"].value_counts().reset_index()
    community_sizes.columns = ["communityId", "community_size"]

    # Join with original dataframe
    df = df.merge(community_sizes, on="communityId", how="left")

    return df

def export_cluster_size_distribution(df, output_csv_path=None):
    """
    Produce a CSV with columns 'community_size' and '#communities'
    from the DataFrame produced by `leiden_user_communities`.

    Args:
        df (pandas.DataFrame): DataFrame returned by `leiden_user_communities`
            (must contain 'community_size').
        output_csv_path (str, optional): output file path.

    Returns:
        pandas.DataFrame: DataFrame with columns 'community_size' and '#communities'.
    """
    if "community_size" not in df.columns:
        raise ValueError("Il DataFrame deve contenere la colonna 'community_size'.")

    # Consider each community only once (in case the df contains multiple rows per community)
    communities = df[["communityId", "community_size"]].drop_duplicates(subset="communityId")

    # Count how many communities for each size and get a DataFrame with the original columns
    dist = communities.groupby("community_size").size().reset_index(name="#communities")

    # Sort by community_size and save CSV (do not rename columns)
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
    
    Args:
        df: DataFrame returned by leiden_user_communities (must contain 'user_id', 'communityId', 'community_size')
        output_csv_path: output file path
        descending: True to order largest clusters first
    
    Returns:
        DataFrame ordered by community_size.
    """
    required = {"user_id", "communityId", "community_size"}
    if not required.issubset(df.columns):
        raise ValueError(f"Il DataFrame deve contenere le colonne: {required}")

    # Ensure we have one row per user_id with the corresponding cluster and size
    users = df[["user_id", "communityId", "community_size"]].drop_duplicates(subset=["user_id"])

    # Sort by cluster size (descending by default) and by communityId for stability
    users = users.sort_values(by=["community_size", "communityId"], ascending=[not descending, True])

    # Save CSV with header: user_id, communityId, community_size
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
    
    Args:
        driver: Neo4j driver.
        df: DataFrame returned by leiden_user_communities (must contain 'user_id', 'communityId')
        max_communities: maximum number of top communities to analyze
        output_csv_path: output file path
    
    Returns:        
        DataFrame with the analysis results.
    """
    community_data = []

    with driver.session() as session:     
        community_groups = df.groupby("communityId")
        # Sort by size descending
        top_communities = community_groups.size().sort_values(ascending=False).head(max_communities).index

        for comm_id in top_communities:
            group = df[df["communityId"] == comm_id]
            if len(community_data) >= max_communities:
                break

            users = group["user_id"].tolist()

            # --- 1. Top users by retweet count in the community ---
            top_users_query = """
            MATCH (u:User)<-[:RETWEETED_FROM]-(v:User)
            WHERE u.user_id IN $users AND v.user_id IN $users
            RETURN u.user_id AS user, count(*) AS retweet_count
            ORDER BY retweet_count DESC
            LIMIT 3
            """
            top_users = session.run(top_users_query, users=users).data()
            top_users_str = ", ".join([f"{r['user']} ({r['retweet_count']})" for r in top_users])

            # --- 2. Community tweets ---
            tweets_query = """
            MATCH (u:User)-[:CREATES|RETWEET]->(t:Tweet)
            WHERE u.user_id IN $users AND t.text IS NOT NULL
            RETURN t.text AS text, t.tweet_label AS label
            """
            tweets = session.run(tweets_query, users=users).data()

            texts = [t["text"] for t in tweets if t["text"]]
            labels = [t["label"] for t in tweets if t["label"]]

            # Distribution of classes
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
            # Extract keywords from community tweets
            keywords = extract_keywords(texts, top_k=10)


            community_data.append({
                "community_id": comm_id,
                "num_users": len(users),
                "top_users": top_users_str,
                "dominant_label": dominant_label,
                "label_distribution": label_distribution,
                "keywords": keywords
            })

    # Put everything into a DataFrame and sort by size
    df_analysis = pd.DataFrame(community_data)
    df_analysis = df_analysis.sort_values("num_users", ascending=False).reset_index(drop=True)

    if output_csv_path:
        df_analysis.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Analisi community salvata in {output_csv_path}")

    return df_analysis

def preprocess_texts(texts):
    """Cleans and lemmatizes texts with spaCy.
    
    Args:
        texts: list of strings
        
    Returns: cleaned texts
    """
    # Load spaCy model (choose en_core_web_sm or it_core_news_sm)
    nlp = spacy.load("en_core_web_sm")  
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
    """Extracts keywords using TF-IDF.
    
    Args:
        texts: list of strings
        top_k: number of top keywords to return
    
    Returns:
        List of top_k keywords
    """
    if not texts:
        return []
    cleaned = preprocess_texts(texts)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(cleaned)
    scores = X.sum(axis=0).A1 # type: ignore
    terms = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[::-1][:top_k]
    return [terms[i] for i in top_indices if terms[i]!="url"]
