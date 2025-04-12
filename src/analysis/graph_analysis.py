from neo4j import GraphDatabase
import logging

def basic_statistics(driver):
    """Restituisce statistiche di base sul grafo."""
    query = """
    MATCH (t:Tweet)
    RETURN count(t) AS num_tweets
    """
    with driver.session() as session:
        result = session.run(query)
        num_tweets = result.single()["num_tweets"]
    
    query = """
    MATCH (u:User)
    RETURN count(u) AS num_users
    """
    with driver.session() as session:
        result = session.run(query)
        num_users = result.single()["num_users"]
    
    query = """
    MATCH (:User)-[r:RETWEET]->(:Tweet)
    RETURN count(r) AS num_retweets
    """
    with driver.session() as session:
        result = session.run(query)
        num_retweets = result.single()["num_retweets"]
    
    return {
        "num_tweets": num_tweets,
        "num_users": num_users,
        "num_retweets": num_retweets
    }

def find_influencers(driver, top_n=10):
    """Trova gli utenti con il maggior numero di retweet."""
    query = """
    MATCH (u:User)-[:RETWEET]->(t:Tweet)
    RETURN u.user_id AS user, count(t) AS retweet_count
    ORDER BY retweet_count DESC
    LIMIT $top_n
    """
    with driver.session() as session:
        result = session.run(query, top_n=top_n)
        return [{"user": record["user"], "retweet_count": record["retweet_count"]} for record in result]


def analyze_diffusion_patterns(driver, tweet_id):
    """Analizza la diffusione di un tweet specifico."""
    query = """
    MATCH path = (u:User)-[r:RETWEET*]->(t:Tweet {tweet_id: $tweet_id})
    RETURN path
    """
    with driver.session() as session:
        result = session.run(query, tweet_id=tweet_id)
        return [record["path"] for record in result]

def get_most_retweeted_tweet(driver):
    query = """
    MATCH (t:Tweet)<-[r:RETWEET]-()
    RETURN t.tweet_id AS tweet_id, COUNT(r) AS retweet_count
    ORDER BY retweet_count DESC
LIMIT 1

    """
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        return record["tweet_id"] if record else None

    tweet_id = get_most_retweeted_tweet(driver)
    if tweet_id:
        logging.info(f"Analisi della diffusione per il tweet {tweet_id}...")
        diffusion = analyze_diffusion_patterns(driver, tweet_id)
        logging.info(f"Diffusione per il tweet {tweet_id}: {diffusion}")
    else:
        logging.warning("Nessun tweet trovato per l'analisi.")

def create_gds_graph(driver):
    """Crea un grafo GDS chiamato 'myGraph' se non esiste già."""
    # Query per verificare se il grafo esiste
    query_drop = "CALL gds.graph.exists('myGraph') YIELD exists"
    query_create = """
    CALL gds.graph.project.cypher(
        'myGraph',
        'MATCH (u:User) RETURN id(u) AS id',
        'MATCH (u1:User)-[r:CREATES|RETWEET|QUOTE|INTERACTION]->(u2:User)
         RETURN id(u1) AS source, id(u2) AS target'
    )
    """
    with driver.session() as session:
        # Verifica se il grafo esiste già
        exists = session.run(query_drop).single()['exists']
        if exists:
            # Se il grafo esiste, lo eliminiamo prima di crearne uno nuovo
            session.run("CALL gds.graph.drop('myGraph')")
        
        # Creiamo il grafo GDS
        session.run(query_create)

    with driver.session() as session:
        exists = session.run(query_drop).single()['exists']
        if exists:
            session.run("CALL gds.graph.drop('myGraph')")
        session.run(query_create)

def compute_pagerank(driver, top_n=10):
    """Calcola il PageRank degli utenti per trovare i più influenti nel grafo."""
    query = """
    CALL gds.pageRank.stream('myGraph')
    YIELD nodeId, score
    RETURN nodeId, score
    ORDER BY score DESC
    LIMIT $top_n
    """
    with driver.session() as session:
        result = session.run(query, {"top_n": top_n})
        users = []
        for record in result:
            node_id = record["nodeId"]
            score = record["score"]
            
            # Verifica se esiste un nodo per l'ID utente
            user_node = session.run("MATCH (u:User) WHERE id(u) = $node_id RETURN u.user_id AS user_id", {"node_id": node_id}).single()
            if user_node:
                users.append({"user": user_node["user_id"], "score": score})
            else:
                users.append({"user": None, "score": score})  # Se non c'è il nodo, segnalalo come None

        return users
