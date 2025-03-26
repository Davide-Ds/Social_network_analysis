import os
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)

def load_tweets_and_labels(path_tweets, path_labels):
    """
    Carica i tweet da source_tweets.txt e li aggiorna con le etichette da label.txt.
    Formato source_tweets.txt: tweet_id <tab> text_content
    Formato label.txt: tweet_id <tab> label
    """
    tweets = {}
    with open(path_tweets, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tweet_id, text = parts
                tweets[tweet_id] = {'text_content': text}
            else:
                logging.warning(f"Dati malformati nella riga dei tweet: {line.strip()}")
                
    with open(path_labels, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                label, tweet_id  = parts
                if tweet_id in tweets:
                    tweets[tweet_id]['label'] = label
                else:
                    logging.warning(f"Tweet {tweet_id} presente in label.txt ma non in source_tweets.txt")
            else:
                logging.warning(f"Dati malformati nella riga delle etichette: {line.strip()}")
    
    return tweets

def process_tree_files(path_tree_files, tweets):
    """
    Processa ogni file nella cartella tree.
    Per ogni file:
      - Usa il nome del file (senza .txt) come tweet_id.
      - Legge la prima riga per estrarre il creatore e aggiorna il dizionario tweets aggiungendo 'created_by'.
      - Per le righe successive, estrae le relazioni RETWEET: (retweeter, tweet_id, creation_delay)
    Restituisce una lista di tuple (retweeter, tweet_id, creation_delay).
    """
    retweet_relations = []
    for filename in os.listdir(path_tree_files):
        file_path = os.path.join(path_tree_files, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            tweet_id = filename.replace(".txt", "")
            logging.info(f"Processo file tree: {file_path} per tweet {tweet_id}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    continue
                # Estrai il creatore dalla prima riga
                first_line = lines[0].strip()
                if "->" in first_line:
                    try:
                        _, right = first_line.split("->")
                        right_values = right.replace("[", "").replace("]", "").split(", ")
                        if len(right_values) >= 2:
                            creator_id = right_values[0].strip("'")
                            if tweet_id in tweets:
                                tweets[tweet_id]['created_by'] = creator_id
                    except Exception as e:
                        logging.warning(f"Errore nel parsing della prima riga in {file_path}: {e}")
                else:
                    logging.warning(f"Prima riga malformata in {file_path}: {first_line}")
                
                # Per le righe successive, estrai le relazioni RETWEET
                for line in lines[1:]:
                    if "->" not in line:
                        logging.warning(f"Riga malformata in {file_path}: {line.strip()}")
                        continue
                    try:
                        _, right = line.strip().split("->")
                        right_values = right.replace("[", "").replace("]", "").split(", ")
                        if len(right_values) < 3:
                            logging.warning(f"Riga malformata in {file_path}: {line.strip()}")
                            continue
                        retweeter = right_values[0].strip("'")
                        creation_delay = float(right_values[2].strip("'"))
                        retweet_relations.append((retweeter, tweet_id, creation_delay))
                    except Exception as e:
                        logging.warning(f"Errore nel parsing della riga in {file_path}: {e}")
    logging.info(f"Processati {len(retweet_relations)} retweet dalle tree files.")
    return retweet_relations

def import_tweet_nodes(driver, tweets):
    with driver.session() as session:
        for tweet_id, data in tweets.items():
            if data.get('label') in ['true', 'non-rumor', 'false', 'unverified']:
                try:
                    # Aggiorna il nodo Tweet con tutte le proprietà
                    session.run(
                        """
                        MERGE (t:Tweet {tweet_id: $tweet_id})
                        SET t.text = $text, t.tweet_label = $label, t.created_by = $created_by
                        """,
                        tweet_id=tweet_id,
                        text=data['text_content'],
                        label=data['label'],
                        created_by=data.get('created_by', "Sconosciuto")
                    )
                    logging.info(f"Nodo Tweet {tweet_id} creato/aggiornato con created_by={data.get('created_by', 'Sconosciuto')}")
                    
                    # Crea la relazione CREATES
                    session.run(
                        """
                        MERGE (u:User {user_id: $created_by})
                        WITH u MATCH (t:Tweet {tweet_id: $tweet_id})
                        MERGE (u)-[:CREATES]->(t)
                        """,
                        created_by=data.get('created_by', "Sconosciuto"),
                        tweet_id=tweet_id
                    )
                    logging.info(f"Relazione CREATES creata: User {data.get('created_by', 'Sconosciuto')} -> Tweet {tweet_id}")
                except Exception as e:
                    logging.error(f"Errore durante l'elaborazione del tweet {tweet_id}: {e}")

def import_retweets(driver, retweet_relations, batch_size=1000):
    logging.info("Importazione delle relazioni RETWEET...")
    batch = []
    with driver.session() as session:
        for relation in retweet_relations:
            user_id, tweet_id, creation_delay = relation
            batch.append((user_id, tweet_id, creation_delay))
            if len(batch) >= batch_size:
                _process_batch(session, batch)
                batch = []
        if batch:
            _process_batch(session, batch)

def _process_batch(session, batch):
    query = """
    UNWIND $batch AS row
    MERGE (u:User {user_id: row.user_id})
    MERGE (t:Tweet {tweet_id: row.tweet_id})
    MERGE (u)-[r:RETWEET {delay: row.creation_delay}]->(t)
    """
    session.run(query, batch=[{"user_id": u, "tweet_id": t, "creation_delay": d} for u, t, d in batch])
    logging.info(f"{len(batch)} relazioni RETWEET elaborate.")

def get_most_retweeted_tweet(driver):
    query = """
    MATCH (t:Tweet)<-[r:RETWEET]-()
    RETURN t.tweet_id AS tweet_id, COUNT(r) AS num_retweets
    ORDER BY num_retweets DESC
    LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        if record:
            return record["tweet_id"]
        else:
            return None
