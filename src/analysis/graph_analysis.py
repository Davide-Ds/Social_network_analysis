import logging
import pandas as pd
logging.getLogger("neo4j").setLevel(logging.CRITICAL)


def basic_statistics(driver):
    """Compute basic statistics of the graph: number of tweets, users, retweets.
    
    Args:
        driver: Neo4j driver.
        
    Returns:
        dict with num_tweets, num_users, num_retweets.    
    """
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
    """Find the users who retweet the most.
    
    Args:
        driver: Neo4j driver.
        top_n: number of top users to return.
         
    Returns:
        List of dicts with user_id and retweet_count.
     """
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
    Find the users with the highest number of distinct users who retweeted them.
    
    Args:
        driver: Neo4j driver.
        top_n: number of top users to return.
    
    Returns:
        List of dicts with user_id and num_retweeters.    
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
    """Analyze the diffusion of a specific tweet by level.
    
    Args:
        driver: Neo4j driver.
        tweet_id: ID of the tweet to analyze.
    
    Returns:
        List of dicts with hop_level, num_users_at_level, users_at_level, paths_at_level.
    """
    query = """
    // 1. Find the Tweet node and get the author
    MATCH (t:Tweet {tweet_id: $tweet_id})
    WITH t.created_by AS sourceUserId, $tweet_id AS tweetId

    // 2. Find the author user node
    MATCH (source:User {user_id: sourceUserId})

    // 3. Spanning tree from source on RETWEETED_FROM relationships
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

    // 4. Filter the paths where all relationships are for that tweet
    WHERE ALL(rel IN relationships(path) WHERE rel.tweet_id = tweetId)

    // 5. Take only the user at the end of the path (exactly at the current level)
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
            logging.error(f"Error occurred while analyzing diffusion patterns: {e}")
            return []


def get_most_retweeted_tweet(driver):
    """
    Returns the tweet_id of the most retweeted tweet.
    
    Args:
        driver: Neo4j driver.
    
    Returns:
        tweet_id of the most retweeted tweet or None if no tweets found.
    """
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


def create_User_gds_graph(driver, graph_name="userGraph_4analysis"):
    """
    Creates a GDS (Graph Data Science) graph named by graph_name if it
    doesn't already exist. Natural orientation uses the direction in the db,
    better for PageRank.

    Args:
        driver: Neo4j driver.
        graph_name (str): name to use for the projected GDS graph.
    """
    query_check = "CALL gds.graph.exists('%s') YIELD exists" % graph_name
    query_create = (
        """
    CALL gds.graph.project(
        '""" + graph_name + """',
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
    )
    with driver.session() as session:
        # Check if the graph already exists
        result = session.run(query_check)
        record = result.single()
        exists = bool(record['exists']) if record is not None and 'exists' in record else False
        if not exists:
            # Create the GDS graph only if it doesn't exist
            session.run(query_create)


def create_complete_gds_graph(driver, graph_name='fullGraph_analysis'):
    """
    Creates a complete GDS graph with the given name containing:
      - User nodes (property: pagerank)
      - Tweet nodes (property: text_embedding)
      - Relationships: CREATES, RETWEET, RETWEETED_FROM, QUOTE, INTERACTION.

    Args:
        driver: Neo4j driver.
        graph_name: name for the projected full graph (default 'fullGraph').
    """
    query_drop = "CALL gds.graph.drop('%s', false) YIELD graphName" % graph_name
    query_create = (
        """
    CALL gds.graph.project(
        '""" + graph_name + """',
        {
            User: {
                properties: ['pagerank']
            },
            Tweet: {
                properties: ['text_embedding']
            }
        },
        {
            CREATES: {type: 'CREATES', orientation: 'NATURAL'},
            RETWEET: {type: 'RETWEET', orientation: 'NATURAL'},
            RETWEETED_FROM: {type: 'RETWEETED_FROM', orientation: 'NATURAL'},
            QUOTE: {type: 'QUOTE', orientation: 'NATURAL'},
            INTERACTION: {type: 'INTERACTION', orientation: 'NATURAL'}
        }
    )
    """
    )
    with driver.session() as session:
        try:
            session.run(query_drop)
        except Exception:
            pass
        session.run(query_create)
    print(f"GDS graph '{graph_name}' created with User (pagerank) and Tweet (text_embedding).")
    

def compute_pagerank(driver, top_n=10, graph_name='userGraph_4analysis'):
    """
    Calculates the PageRank of users, saves it as the 'pagerank' property in User nodes,
    and returns the top_n most influential users.

    Args:
        driver: Neo4j driver.
        top_n: number of top users to return.
        graph_name: name of the GDS graph to use for PageRank.

    Returns:
        List of dicts with user_id and pagerank score.
    """
    # Compute and write the PageRank in User nodes as property 'pagerank'
    write_query = "CALL gds.pageRank.write('%s', { writeProperty: 'pagerank' }) YIELD nodePropertiesWritten, ranIterations" % graph_name
    # Retrieve the top_n users by PageRank
    select_query = """
    MATCH (u:User)
    RETURN u.user_id AS user, u.pagerank AS score
    ORDER BY score DESC
    LIMIT $top_n
    """
    with driver.session() as session:
        session.run(write_query)
        result = session.run(select_query, {"top_n": top_n})
        return [{"user": record["user"], "score": record["score"]} for record in result]


def get_top_fake_news_creators(driver, top_n=10):
    """
    Returns the top fake news creators with details 
    about their tweets and retweets.
    
    Args:
        driver: Neo4j driver.
        top_n: number of top creators to return.
    
    Returns:
        List of top fake news creators with their tweet details.
    """
    query = """
    MATCH (u:User)-[:CREATES]->(t:Tweet)
    WITH u, 
         collect(t) AS all_tweets,
         [tweet IN collect(t) WHERE tweet.tweet_label = 'false' | tweet.tweet_id] AS fake_tweet_ids

    WITH 
      u, 
      fake_tweet_ids,
      size(fake_tweet_ids) AS num_fake_tweets,
      size(all_tweets) AS total_tweets

    OPTIONAL MATCH (other:User)-[r:RETWEETED_FROM]->(target:User)
    WHERE r.tweet_id IN fake_tweet_ids

    RETURN 
      u.user_id AS user_id,
      total_tweets,
      num_fake_tweets,
      COUNT(r) AS total_retweets_on_fake,
      fake_tweet_ids
    ORDER BY num_fake_tweets DESC, total_retweets_on_fake DESC
    LIMIT $top_n
    """
    with driver.session() as session:
        try:
            result = session.run(query, {"top_n": top_n})
            top_creators = []
            for record in result:
                creator_data = {
                    "user_id": record["user_id"],
                    "total_tweets": record["total_tweets"],
                    "num_fake_tweets": record["num_fake_tweets"],
                    "total_retweets_on_fake": record["total_retweets_on_fake"],
                    "fake_tweet_ids": record["fake_tweet_ids"],
                }
                top_creators.append(creator_data)
            return top_creators
        except Exception as e:
            logging.error(f"Errore durante l'esecuzione della query: {e}")
            return []

def get_class_stats(driver):
    """
    Compute aggregate statistics for tweet classes 
    (only tweets with non-null labels).
    
    Args:
        driver: Neo4j driver.
    
    Returns:
        DataFrame with aggregate statistics for each tweet class.
    """
    query = """
    MATCH (t:Tweet)
    WHERE t.tweet_label IS NOT NULL
    OPTIONAL MATCH (u:User)-[r:RETWEET]->(t)
    WITH t.tweet_label AS class, t, count(u) AS retweet_count, coalesce(max(r.delay), 0) AS max_delay
    RETURN 
        class,
        count(DISTINCT t) AS num_tweets,
        sum(retweet_count) AS total_retweets,
        round(toFloat(sum(retweet_count)) / count(DISTINCT t), 2) AS avg_retweets_per_tweet,
        round(sum(max_delay)/60, 2) AS total_propagation_hours,
        round(toFloat(sum(max_delay)) / count(DISTINCT t)/60, 2) AS avg_propagation_hours_per_tweet
    ORDER BY num_tweets DESC
    """
    with driver.session() as session:
        result = session.run(query)
        records = [dict(record) for record in result]
        df = pd.DataFrame(records)
        return df

