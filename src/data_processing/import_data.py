import os
import logging
import re

# Set logging level to WARNING to disable lower-level logs
logging.basicConfig(level=logging.WARNING)

def load_tweets_and_labels(path_tweets, path_labels):
    """
    Load tweets from source_tweets.txt and update them with labels from label.txt.
    source_tweets.txt format: tweet_id <tab> text_content
    label.txt format: label:tweet_id
    """
    tweets = {}
    with open(path_tweets, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tweet_id, text = parts
                tweets[tweet_id] = {'text_content': text}
            else:
                logging.warning(f"Malformatted data in tweet line: {line.strip()}")
                
    with open(path_labels, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                label, tweet_id = parts
                if tweet_id in tweets:
                    tweets[tweet_id]['label'] = label
                else:
                    logging.warning(f"Tweet {tweet_id} present in label.txt but not in source_tweets.txt")
            else:
                logging.warning(f"Malformatted data in label line: {line.strip()}")
    
    return tweets

def classify_relation(S_user, S_tweet, S_time, p_user, p_tweet, p_time, c_user, c_tweet, c_time):
    """
    Classify the relation according to described cases.

    - Case 1 (RETWEET_L1): if the parent node is the source S and the child has the same tweetId (c_time > S_time, c_user != S_user)
    - Case 2 (QUOTE): if the parent node is S and the child has a different tweetId
    - Case 3 (RETWEET_LN): if the parent is not S, but both have the source's tweetId and c_time > p_time
    - Case 4 (TIME_ANOMALY): if c_time < p_time
    - Case 6 (INTERACTION_RETURN_TO_SOURCE): if the parent has a different tweetId and the child has the source's tweetId
    - Case 7 (INTERACTION_PARENT_OF_PARENT): a node is parent of a node that was previously its parent. user_ids and tweetIDs different, p_time > c_time
    - All other cases are classified as INTERACTION_OTHERS.
    """

    relation = "Not Identified"
    
    if c_time < p_time:            #case 4: anomalia tempo, il nodo figlio è nato prima di suo padre
        relation = "INTERACTION"   
    elif p_user == S_user and p_tweet == S_tweet and abs(p_time - S_time) < 1e-6: # Il nodo padre è il source S
        if c_tweet == S_tweet:
            if c_user != S_user and c_time > S_time:
                relation = "RETWEET"   # Caso 1: RETWEET_L1
            else:
                relation = "INTERACTION"
        else:
            relation = "QUOTE"         # Caso 2: QUOTE
    else:  # Nodo padre non è S
        if p_tweet == S_tweet and c_tweet == S_tweet and c_time > p_time:
            relation = "RETWEET"       # Caso 3: RETWEET (livello >1)
        elif p_tweet != S_tweet and c_tweet == S_tweet:
            relation = "INTERACTION"   # Caso 6: Ritorno al source tweetID
        elif p_tweet == S_tweet and c_tweet != S_tweet:
            relation = "QUOTE"   # citazione di un retweet
        elif p_tweet != S_tweet and c_tweet != p_tweet and c_time > p_time:
            relation = "QUOTE"   # citazione di una citazione (Quote_Ln)
        else:
            relation = "INTERACTION"   # Altri casi ambigui incluso Caso 7: Padre del padre
    
    return relation   

def process_tree_files(path_tree_files, tweets):
    """
    Process each file in the tree folder.
    For each file:
      - Use the filename (without .txt) as tweet_id.
      - Read the first line to extract the creator and update tweets dict with 'created_by'.
      - For following lines, extract relations and classify them using classify_relation.
    Returns a list of tuples:
      (p_user, p_tweet, p_time, c_user, c_tweet, c_time, relation_type)
    """
    retweet_relations = []
    pattern = re.compile(r"\['([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\]\s*->\s*\['([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\]")
    
    for filename in os.listdir(path_tree_files):
        file_path = os.path.join(path_tree_files, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            tweet_id = filename.replace(".txt", "")
            logging.info(f"Processing tree file: {file_path} for tweet {tweet_id}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    continue
                # La prima riga definisce il source S
                header = pattern.match(lines[0].strip())
                if header:
                    # Ignoriamo la parte sinistra e usiamo la parte destra per definire S
                    _, _, _, S_user, S_tweet, S_time_str = header.groups()
                    try:
                        S_time = float(S_time_str)
                    except ValueError:
                        S_time = 0.0
                    if tweet_id in tweets:
                        tweets[tweet_id]['created_by'] = S_user
                else:
                    logging.warning(f"First line malformed in {file_path}: {lines[0].strip()}")
                    continue
                
                # Processa le righe successive
                for line in lines[1:]:
                    header = pattern.match(line.strip())
                    if not header:
                        logging.warning(f"Line malformed in {file_path}: {line.strip()}")
                        continue
                    p_user, p_tweet, p_time_str, c_user, c_tweet, c_time_str = header.groups()
                    try:
                        p_time = float(p_time_str)
                        c_time = float(c_time_str)
                    except ValueError:
                        logging.warning(f"Time conversion error in {file_path} line: {line.strip()}")
                        continue

                    rel_type = classify_relation(S_user, S_tweet, S_time, p_user, p_tweet, p_time, c_user, c_tweet, c_time)
                    retweet_relations.append((p_user, p_tweet, p_time, c_user, c_tweet, c_time, rel_type))
    logging.info(f"Processed {len(retweet_relations)} retweets from tree files.")
    return retweet_relations

def import_tweet_nodes(driver, tweets):
    """
    Use a session to create Tweet nodes and CREATES relationships
    """
    with driver.session() as session:
        for tweet_id, data in tweets.items():
            if data.get('label') in ['true', 'non-rumor', 'false', 'unverified']:
                try:
                    session.run(
                        """
                        MERGE (t:Tweet {tweet_id: $tweet_id})
                        SET t.text = $text, t.tweet_label = $label, t.created_by = $created_by
                        """,
                        tweet_id=tweet_id,
                        text=data['text_content'],
                        label=data['label'],
                        created_by=data.get('created_by', "Unknown")
                    )
                    logging.info(f"Tweet node {tweet_id} created/updated with created_by={data.get('created_by', 'Unknown')}")
                    
                    # Crea la relazione CREATES
                    session.run(
                        """
                        MERGE (u:User {user_id: $created_by})
                        WITH u MATCH (t:Tweet {tweet_id: $tweet_id})
                        MERGE (u)-[:CREATES]->(t)
                        """,
                        created_by=data.get('created_by', "Unknown"),
                        tweet_id=tweet_id
                    )
                    logging.info(f"CREATES relationship created: User {data.get('created_by', 'Unknown')} -> Tweet {tweet_id}")
                except Exception as e:
                    logging.error(f"Error processing tweet {tweet_id}: {e}")

def import_retweets(driver, retweet_relations, batch_size=1000):
    logging.info("Importing relations (RETWEET, QUOTE, INTERACTION, etc.) into Neo4j...")
    batch = []
    user_batch = []
    existing_tweet_ids = set()

    with driver.session() as session:
        for relation in retweet_relations:
            # relation: (p_user, p_tweet, p_time, c_user, c_tweet, c_time, rel_type)
            p_user, p_tweet_id, _, c_user, c_tweet_id, c_time, rel_type = relation
            batch.append((c_user, p_tweet_id, c_time, rel_type))  # add parent tweet id to main batch
            # Create a user-to-user relation only for RETWEET, including the retweeted tweet id
            if rel_type == "RETWEET":
                user_batch.append((c_user, p_user, p_tweet_id))
            existing_tweet_ids.add(p_tweet_id)
            if rel_type == 'QUOTE' and c_tweet_id not in existing_tweet_ids:
                batch.append((c_user, c_tweet_id, c_time, "CREATES"))
                existing_tweet_ids.add(c_tweet_id)
            if len(batch) >= batch_size:
                _process_batch(session, batch)
                batch = []
            if len(user_batch) >= batch_size:
                _process_user_batch(session, user_batch)
                user_batch = []
        if batch:
            _process_batch(session, batch)
        if user_batch:
            _process_user_batch(session, user_batch)

def _process_batch(session, batch):
    """
    Use APOC to dynamically create relationship types; set created_by on Tweet when relation is CREATES
    """
    query = """
    UNWIND $batch AS row
    MERGE (u:User {user_id: row.user_id})
    MERGE (t:Tweet {tweet_id: row.tweet_id})
    FOREACH (_ IN CASE WHEN row.relation_type = 'CREATES' THEN [1] ELSE [] END |
        SET t.created_by = row.user_id
    )
    WITH u, t, row
    CALL apoc.create.relationship(u, row.relation_type, {delay: row.creation_delay}, t) YIELD rel
    RETURN count(rel) AS count
    """
    session.run(query, batch=[
        {"user_id": u, "tweet_id": t, "creation_delay": d, "relation_type": rt}
        for u, t, d, rt in batch
    ])
    logging.info(f"{len(batch)} relationships imported with dynamic relationship type.")

def _process_user_batch(session, user_batch):
    """
    Create RETWEETED_FROM relationships between users and attach the retweeted tweet id as property
    """
    query = """
    UNWIND $batch AS row
    MERGE (child:User {user_id: row.child_user})
    MERGE (parent:User {user_id: row.parent_user})
    MERGE (child)-[r:RETWEETED_FROM {tweet_id: row.tweet_id}]->(parent)
    """
    session.run(query, batch=[{"child_user": c, "parent_user": p, "tweet_id": t} for c, p, t in user_batch])
    logging.info(f"{len(user_batch)} RETWEETED_FROM relationships imported between users.")

def create_indexes(driver):
    """
    Create indexes on User.user_id and Tweet.tweet_id
    """
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.user_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (t:Tweet) ON (t.tweet_id)")
    logging.info("Indexes created on User(user_id) and Tweet(tweet_id)")