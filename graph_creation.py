import os
import logging
from neo4j import GraphDatabase

# Configurazione del logging
logging.basicConfig(level=logging.INFO)

# Dettagli di connessione a Neo4j
port = os.getenv("NEO4J_PORT", "7687")  # Default 7687 se non è impostata
uri = f"bolt://localhost:{port}"
username = os.getenv("NEO4J_USERNAME", "neo4j")  # Default: neo4j
password = os.getenv("NEO4J_PASSWORD", "password!")  # Default: password!

# Connessione a Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))


def import_data_to_neo4j(driver, tweets, path_tree_files, retweet_limit=50000000):
    with driver.session() as session:
        # Crea nodi per i tweet e gli utenti originali
        for tweet_id, data in tweets.items():
            if data.get('label') in ['true', 'non-rumor','false']: #dds: aggiungere classe non-rumor perchè il tweet è vero anche
                try:
                    # Crea il nodo per il tweet originale
                    session.run(
                        "MERGE (t:Tweet {id: $id, text_content: $text_content, label: $label})",
                        id=tweet_id, text_content=data['text_content'], label=data['label']
                    )
                    logging.info(f"Nodo Tweet {tweet_id} creato o trovato")

                    # Crea il nodo per l'utente che ha creato il tweet e la relazione CREATES
                    if 'user_id' in data:   #dds: ma gli userId non vengono messi nel dict Tweets quando viene creato (ln 105-117)
                        session.run(
                            "MERGE (u:User {id: $user_id}) "
                            "WITH u "
                            "MATCH (t:Tweet {id: $id}) "
                            "MERGE (u)-[:CREATES]->(t)",
                            user_id=data['user_id'], id=tweet_id
                        )
                        logging.info(f"Relazione CREATES creata: User {data['user_id']} -> Tweet {tweet_id}")

                except Exception as e:
                    logging.error(f"Errore durante l'elaborazione del tweet {tweet_id}: {e}")

        # Processo relazioni di retweet dai file tree
        if not os.path.isdir(path_tree_files):
            logging.error(f"Il percorso non esiste: {path_tree_files}")
            return
        i = 0
        for tweet_id in tweets.keys():
            if i >= 1: #dds: per ora solo il primo tweet
                break
            i += 1
            tree_file_path = os.path.join(path_tree_files, f'{tweet_id}.txt')
            if os.path.exists(tree_file_path):
                with open(tree_file_path, 'r', encoding='utf-8') as file_tree:
                    retweet_count = 0
                    for line in file_tree:
                        if retweet_count >= retweet_limit:
                            break
                        try:
                            # Parsing della relazione parent-child
                            parent, child = line.strip().split('->')
                            parent = parent.strip().strip('[]').split(',')
                            child = child.strip().strip('[]').split(',')

                            #parent_user_id = parent[0].strip().strip("'")   #dds: non usato dopo
                            parent_tweet_id = parent[1].strip().strip("'")
                            #parent_retweet_time = parent[2].strip().strip("'")  # Tempo al quale è stato retweettato. dds: sempre 0 per i source tweet, va preso dal nodo figlio

                            child_user_id = child[0].strip().strip("'")
                            #child_tweet_id = child[1].strip().strip("'")  #dds: non usato dopo
                            child_retweet_time = child[2].strip().strip("'")
                                                                       
                            # Verifica se il tweet originale esiste
                            if parent_tweet_id in tweets and tweets[parent_tweet_id].get('label') in ['true','non-rumor', 'false']:
                                # Crea il nodo per l'utente che retweetta
                                session.run(
                                    "MERGE (u:User {id: $user_id})",
                                    user_id=child_user_id
                                )
                                logging.info(f"Nodo User {child_user_id} creato o trovato")

                                # Crea il nodo per il tweet che è stato retweettato. Dds: ma esistono già questi nodi, i source tweets creati prima (ln 24)
                                session.run(
                                    "MERGE (t:Tweet {id: $parent_tweet_id})", # dds: dovrebbe essere il tweet figlio, se tweetId diverso dal padre
                                    parent_tweet_id=parent_tweet_id
                                )
                                logging.info(f"Nodo Tweet {parent_tweet_id} creato o trovato")

                                # Crea la relazione di retweet con il tempo, evitando prodotti cartesiani
                                session.run(
                                    "MATCH (u:User {id: $user_id}) "
                                    "OPTIONAL MATCH (t:Tweet {id: $parent_tweet_id}) "
                                    "WITH u, t "
                                    "MERGE (u)-[:RETWEETS {time: $retweet_time}]->(t)",
                                    user_id=child_user_id, parent_tweet_id=parent_tweet_id, retweet_time=child_retweet_time 
                                )
                                logging.info(f"Relazione RETWEETS creata tra User {child_user_id} e Tweet {parent_tweet_id} con tempo {child_retweet_time}")

                                retweet_count += 1

                        except Exception as e:
                            logging.error(f"Errore durante l'elaborazione della relazione retweet per {parent_tweet_id}: {e}")

                            
# Percorsi dei file
path_source_tweets = 'rumor_detection_2017/twitter16/source_tweets.txt'
path_labels = 'rumor_detection_2017/twitter16/label.txt'
path_tree_files = 'rumor_detection_2017/twitter16/tree/'

# Step 1: Importa Tweet e Etichette
tweets = {}
with open(path_source_tweets, 'r', encoding='utf-8') as file_source_tweets:
    for line in file_source_tweets:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            tweet_id, text_content = parts
            tweets[tweet_id] = {'text_content': text_content}

with open(path_labels, 'r', encoding='utf-8') as file_labels:
    for line in file_labels:
        label, tweet_id = line.strip().split(':')
        if tweet_id in tweets:
            tweets[tweet_id]['label'] = label

#print(tweets)


# Importa i dati in Neo4j
import_data_to_neo4j(driver, tweets, path_tree_files)

# Chiudi la connessione
driver.close()

