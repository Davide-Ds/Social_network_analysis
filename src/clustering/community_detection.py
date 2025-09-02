import pandas as pd

def leiden_user_communities(driver, graph_name="usersGraph"):
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

def export_cluster_size_distribution(df, output_csv_path="src/clustering/community_size_distribution.csv"):
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
    dist.to_csv(output_csv_path, index=False, encoding="utf-8")
    
    print(f"Distribuzione completa dimensione cluster salvata in {output_csv_path}")
    return dist


def export_users_ordered_by_cluster(df, output_csv_path="src/clustering/users_by_cluster.csv", descending=True):
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
    users.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"Elenco completo utenti per cluster salvato in {output_csv_path}")
    return users