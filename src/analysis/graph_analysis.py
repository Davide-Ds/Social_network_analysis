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
    WITH path, length(path) AS diffusion_length, COLLECT(u.username) AS users_involved
    UNWIND relationships(path) AS retweet
    RETURN path, diffusion_length, users_involved, COLLECT(retweet.created_at) AS retweet_dates
    """
    with driver.session() as session:
        try:
            result = session.run(query, tweet_id=tweet_id)
            diffusion_paths = []
            for record in result:
                # Creazione di un dizionario per ogni percorso, con dettagli sulla diffusione
                path_data = {
                    "path": record["path"],
                    "diffusion_length": record["diffusion_length"],
                    "users_involved": record["users_involved"],
                    "retweet_dates": record["retweet_dates"]
                }
                diffusion_paths.append(path_data)
            return diffusion_paths
        except Exception as e:
            logging.error(f"Errore durante l'esecuzione della query: {e}")
            return None




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


def create_gds_graph(driver):
    """Crea un grafo GDS (Graph Data Science) chiamato 'myGraph' se non esiste già."""
    query_check = "CALL gds.graph.exists('myGraph') YIELD exists"
    query_create = """
    CALL gds.graph.project(
        'myGraph',
        {
            User: {
                label: 'User'
            }
        },
        {
            CREATES: {
                type: 'CREATES',
                orientation: 'UNDIRECTED'
            },
            RETWEET: {
                type: 'RETWEET',
                orientation: 'UNDIRECTED'
            },
            QUOTE: {
                type: 'QUOTE',
                orientation: 'UNDIRECTED'
            },
            INTERACTION: {
                type: 'INTERACTION',
                orientation: 'UNDIRECTED'
            }
        }
    )
    """
    with driver.session() as session:
        # Verifica se il grafo esiste già
        exists = session.run(query_check).single()['exists']
        if not exists:
            # Creiamo il grafo GDS solo se non esiste
            session.run(query_create)


def compute_pagerank(driver, top_n=10):
    """Calcola il PageRank degli utenti per trovare i più influenti nel grafo."""
    query = """
    CALL gds.pageRank.stream('myGraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).user_id AS user, score
    ORDER BY score DESC
    LIMIT $top_n
    """
    with driver.session() as session:
        result = session.run(query, {"top_n": top_n})
        return [{"user": record["user"], "score": record["score"]} for record in result]
