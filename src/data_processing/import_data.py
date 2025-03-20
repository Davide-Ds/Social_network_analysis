import os
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)

def load_tweets_and_labels(path_source_tweets, path_labels):
    tweets = {}
    with open(path_source_tweets, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                tweet_id, text_content = parts
                tweets[tweet_id] = {'text_content': text_content}
    
    with open(path_labels, 'r', encoding='utf-8') as f:
        for line in f:
            label, tweet_id = line.strip().split(':')
            if tweet_id in tweets:
                tweets[tweet_id]['label'] = label
    return tweets

def import_tweet_nodes(driver, tweets):
    """
    Importa i nodi Tweet e, se presenti, la relazione CREATES per gli autori.
    """
    with driver.session() as session:
        for tweet_id, data in tweets.items():
            if data.get('label') in ['true', 'non-rumor','false']:
                try:
                    # Crea il nodo per il tweet originale
                    session.run(
                        "MERGE (t:Tweet {id: $id, text_content: $text_content, label: $label})",
                        id=tweet_id, text_content=data['text_content'], label=data['label']
                    )
                    logging.info(f"Nodo Tweet {tweet_id} creato o trovato")
                    
                    # Se è disponibile l'user_id, crea il nodo utente e la relazione CREATES
                    if 'user_id' in data:
                        session.run(
                            "MERGE (u:User {id: $user_id}) "
                            "WITH u MATCH (t:Tweet {id: $id}) "
                            "MERGE (u)-[:CREATES]->(t)",
                            user_id=data['user_id'], id=tweet_id
                        )
                        logging.info(f"Relazione CREATES creata: User {data['user_id']} -> Tweet {tweet_id}")
                except Exception as e:
                    logging.error(f"Errore durante l'elaborazione del tweet {tweet_id}: {e}")
                    
def load_tweets_and_labels(source_path, labels_path):
    tweets = {}
    with open(source_path, 'r', encoding='utf-8') as file_source:
        for line in file_source:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                tweet_id, text_content = parts
                tweets[tweet_id] = {'text_content': text_content}
    with open(labels_path, 'r', encoding='utf-8') as file_labels:
        for line in file_labels:
            label, tweet_id = line.strip().split(':')
            if tweet_id in tweets:
                tweets[tweet_id]['label'] = label
    return tweets
