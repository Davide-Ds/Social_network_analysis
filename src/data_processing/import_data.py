import os
import logging
import re

# Set logging level to WARNING to disable lower-level logs
logging.basicConfig(level=logging.WARNING)

def load_tweets_and_labels(path_tweets, path_labels):
    """
    Carica i tweet da source_tweets.txt e li aggiorna con le etichette da label.txt.
    Formato source_tweets.txt: tweet_id <tab> text_content
    Formato label.txt: label:tweet_id
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
                label, tweet_id = parts
                if tweet_id in tweets:
                    tweets[tweet_id]['label'] = label
                else:
                    logging.warning(f"Tweet {tweet_id} presente in label.txt ma non in source_tweets.txt")
            else:
                logging.warning(f"Dati malformati nella riga delle etichette: {line.strip()}")
    
    return tweets

def classify_relation(S_user, S_tweet, S_time, p_user, p_tweet, p_time, c_user, c_tweet, c_time):
    """
    Classifica la relazione in base ai casi descritti.

    - Caso 1 (RETWEET_L1): se il nodo padre è il source S e il figlio ha lo stesso tweetId (c_time > S_time, c_user != S_user)
    - Caso 2 (QUOTE): se il nodo padre è S e il figlio ha tweetId diverso
    - Caso 3 (RETWEET_LN): se il nodo padre non è S, ma entrambi hanno il tweetId del source e c_time > p_time
    - Caso 4 (ANOMALIA_TEMPO): se c_time < p_time
    - Caso 6 (INTERACTION_RITORNO_SOURCE): se il nodo padre ha tweetId diverso e il figlio ha tweetId uguale al source
    - Caso 7 (INTERACTION_PADRE_DEL_PADRE): un nodo è padre di un nodo che in precedenza era suo padre. User_ids e tweetIDs diversi, p_time > c_time
    - Tutti gli altri casi sono classificati come INTERACTION_OTHERS.
    """

    relation = "Not Identified"
    
    if c_time < p_time:            #caso 4: anomalia tempo, il nodo figlio è nato prima di suo padre
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
            relation = "QUOTE"   # citazione di un ritweet
        elif p_tweet != S_tweet and c_tweet != p_tweet and c_time > p_time:
            relation = "QUOTE"   # citazione di una citazione (Quote_Ln)
        else:
            relation = "INTERACTION"   # Altri casi ambigui incluso Caso 7: Padre del padre
    
    return relation   

def process_tree_files(path_tree_files, tweets):
    """
    Processa ogni file nella cartella tree.
    Per ogni file:
      - Usa il nome del file (senza .txt) come tweet_id.
      - Legge la prima riga per estrarre il creatore e aggiorna il dizionario tweets con 'created_by'.
      - Per le righe successive, estrae le relazioni e le classifica usando classify_relation.
    Restituisce una lista di tuple:
      (p_user, p_tweet, p_time, c_user, c_tweet, c_time, relation_type)
    """
    retweet_relations = []
    pattern = re.compile(r"\['([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\]\s*->\s*\['([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\]")
    
    for filename in os.listdir(path_tree_files):
        file_path = os.path.join(path_tree_files, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            tweet_id = filename.replace(".txt", "")
            logging.info(f"Processo file tree: {file_path} per tweet {tweet_id}")
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
                    logging.warning(f"Prima riga malformata in {file_path}: {lines[0].strip()}")
                    continue
                
                # Processa le righe successive
                for line in lines[1:]:
                    header = pattern.match(line.strip())
                    if not header:
                        logging.warning(f"Linea malformata in {file_path}: {line.strip()}")
                        continue
                    p_user, p_tweet, p_time_str, c_user, c_tweet, c_time_str = header.groups()
                    try:
                        p_time = float(p_time_str)
                        c_time = float(c_time_str)
                    except ValueError:
                        logging.warning(f"Errore conversione tempi in {file_path} linea: {line.strip()}")
                        continue

                    rel_type = classify_relation(S_user, S_tweet, S_time, p_user, p_tweet, p_time, c_user, c_tweet, c_time)
                    retweet_relations.append((p_user, p_tweet, p_time, c_user, c_tweet, c_time, rel_type))
    logging.info(f"Processati {len(retweet_relations)} retweet dalle tree files.")
    return retweet_relations

def import_tweet_nodes(driver, tweets):
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
    logging.info("Importazione delle relazioni (RETWEET, QUOTE, INTERACTION, ecc.) in Neo4j...")
    batch = []
    existing_tweet_ids = set()

    with driver.session() as session:
        for relation in retweet_relations:
            # relation: (p_user, p_tweet, p_time, c_user, c_tweet, c_time, rel_type)
            _, p_tweet_id, _, c_user, c_tweet_id, c_time, rel_type = relation 
            batch.append((c_user, p_tweet_id, c_time, rel_type))   # mettete il tweet padre come tweet_id
            existing_tweet_ids.add(p_tweet_id)
            if rel_type=='QUOTE' and c_tweet_id not in existing_tweet_ids: #TODO:mettere controllo su tipo relazione, se quote allora aggiungi crazione
                batch.append((c_user, c_tweet_id, c_time, "CREATES"))  # crea un nuovo tweet per il quote del figlio
                existing_tweet_ids.add(c_tweet_id)
            if len(batch) >= batch_size:
                _process_batch(session, batch)
                batch = []
        if batch:
            _process_batch(session, batch)

def _process_batch(session, batch):
    # Usiamo APOC per creare dinamicamente il tipo di relazione. Se il tweet è nuovo aggiunge il creatore
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
    session.run(query, batch=[{"user_id": u, "tweet_id": t, "creation_delay": d, "relation_type": rt} for u, t, d, rt in batch])
    logging.info(f"{len(batch)} relazioni importate con tipo di relazione dinamico.")

def create_indexes(driver):
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.user_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (t:Tweet) ON (t.tweet_id)")
    logging.info("Indici creati su User(user_id) e Tweet(tweet_id)")

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