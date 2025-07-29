import logging


def basic_statistics(driver):
    """Restituisce statistiche di base sul grafo."""
    stats_queries = {
        "num_tweets": "MATCH (t:Tweet) RETURN count(t) AS value",
        "num_users": "MATCH (u:User) RETURN count(u) AS value",
        "num_retweets": "MATCH (:User)-[r:RETWEET]->(:Tweet) RETURN count(r) AS value"
    }
    stats = {}
    with driver.session() as session:
        for key, query in stats_queries.items():
            result = session.run(query)
            stats[key] = result.single()["value"]
    return stats

def find_frequent_retwetters(driver, top_n=10):
    """Trova gli utenti che retweettano maggiormente."""
    query = """
    MATCH (u:User)-[:RETWEET]->(t:Tweet)
    RETURN u.user_id AS user, count(t) AS retweet_count
    ORDER BY retweet_count DESC
    LIMIT $top_n
    """
    with driver.session() as session:
        result = session.run(query, top_n=top_n)
        return [{"user": record["user"], "retweet_count": record["retweet_count"]} for record in result]

def find_most_retweeted_users(driver, top_n=10):
    """
    Trova gli n utenti con il maggior numero di utenti distinti che li hanno retweettati.
    """
    query = """
    MATCH (u:User)<-[:RETWEETED_FROM]-(r:User)
    WITH u, COUNT(DISTINCT r) AS num_retweeters
    ORDER BY num_retweeters DESC
    LIMIT 20
    RETURN u.user_id AS user, num_retweeters
    """
    with driver.session() as session:
        result = session.run(query, top_n=top_n)
        return [{"user": record["user"], "num_retweeters": record["num_retweeters"]} for record in result]

def analyze_diffusion_patterns(driver, tweet_id):
    """Analizza la diffusione di un tweet specifico per livello."""
    query = """
    // 1. Trova il nodo Tweet e prendi l'autore
    MATCH (t:Tweet {tweet_id: $tweet_id})
    WITH t.created_by AS sourceUserId, $tweet_id AS tweetId

    // 2. Trova il nodo utente autore
    MATCH (source:User {user_id: sourceUserId})

    // 3. Spanning tree da source su relazioni RETWEETED_FROM
    CALL apoc.path.spanningTree(
      source,
      {
        relationshipFilter: "<RETWEETED_FROM",
        labelFilter: "User",
        maxLevel: 20,
        bfs: true,
        filterStartNode: true
      }
    )
    YIELD path

    // 4. Filtra i path dove tutte le relazioni sono per quel tweet
    WHERE ALL(rel IN relationships(path) WHERE rel.tweet_id = tweetId)

    // 5. Prendi solo l'utente all'estremo del path (esattamente al livello corrente)
    WITH length(path) AS hop_level, last(nodes(path)) AS user, path
    WHERE hop_level > 0

    RETURN 
      hop_level,
      count(DISTINCT user) AS num_users_at_level,
      collect(DISTINCT user.user_id) AS users_at_level,
      collect(DISTINCT path) AS paths_at_level
    ORDER BY hop_level
    """
    with driver.session() as session:
        try:
            result = session.run(query, tweet_id=tweet_id)
            diffusion_levels = []
            for record in result:
                level_data = {
                    "hop_level": record["hop_level"],
                    "num_users_at_level": record["num_users_at_level"],
                    "users_at_level": record["users_at_level"],
                    "paths_at_level": record["paths_at_level"]
                }
                diffusion_levels.append(level_data)
            return diffusion_levels
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
    """Crea un grafo GDS (Graph Data Science) chiamato 'myGraph' se non esiste già. Natural orientation usa la direzione nel db, meglio per il page rank"""
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
                orientation: 'NATURAL'      
            },
            RETWEET: {
                type: 'RETWEET',
                orientation: 'NATURAL'
            },
            RETWEETED_FROM: {
                type: 'RETWEETED_FROM',
                orientation: 'NATURAL'
            },
            QUOTE: {
                type: 'QUOTE',
                orientation: 'NATURAL'
            },
            INTERACTION: {
                type: 'INTERACTION',
                orientation: 'NATURAL'
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
