import os
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)

def import_data_to_neo4j(driver, tweets, path_tree_files, retweet_limit=50000000):
    with driver.session() as session:
        # Ciclo per creare nodi tweet e relazioni CREATES (se 'user_id' è presente)
        for tweet_id, data in tweets.items():
            if data.get('label') in ['true', 'non-rumor','false']:
                try:
                    session.run(
                        "MERGE (t:Tweet {id: $id, text_content: $text_content, label: $label})",
                        id=tweet_id, text_content=data['text_content'], label=data['label']
                    )
                    logging.info(f"Nodo Tweet {tweet_id} creato o trovato")
                    if 'user_id' in data:
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
        
        # Elaborazione dei file tree per creare relazioni RETWEETS
        if not os.path.isdir(path_tree_files):
            logging.error(f"Il percorso non esiste: {path_tree_files}")
            return
        i = 0
        for tweet_id in tweets.keys():
            if i >= 1:  # per ora solo il primo tweet
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
                            parent, child = line.strip().split('->')
                            parent = parent.strip().strip('[]').split(',')
                            child = child.strip().strip('[]').split(',')
                            parent_tweet_id = parent[1].strip().strip("'")
                            child_user_id = child[0].strip().strip("'")
                            child_retweet_time = child[2].strip().strip("'")
                            if parent_tweet_id in tweets and tweets[parent_tweet_id].get('label') in ['true','non-rumor', 'false']:
                                session.run(
                                    "MERGE (u:User {id: $user_id})",
                                    user_id=child_user_id
                                )
                                logging.info(f"Nodo User {child_user_id} creato o trovato")
                                session.run(
                                    "MERGE (t:Tweet {id: $parent_tweet_id})",
                                    parent_tweet_id=parent_tweet_id
                                )
                                logging.info(f"Nodo Tweet {parent_tweet_id} creato o trovato")
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

# Lettura dei file
path_source_tweets = 'data/twitter16/source_tweets.txt'
path_labels = 'data/twitter16/label.txt'
path_tree_files = 'data/twitter16/tree/'

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

if __name__ == '__main__':
    from src.utils.neo4j_utils import get_neo4j_driver
    driver = get_neo4j_driver()
    import_data_to_neo4j(driver, tweets, path_tree_files)
    driver.close()
