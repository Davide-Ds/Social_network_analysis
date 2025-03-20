from src.utils.neo4j_utils import get_neo4j_driver
from src.data_processing.import_data import import_data_to_neo4j
import os

def main():
    # Percorsi dei dati (puoi rendere questi parametri configurabili)
    path_source_tweets = os.path.join("data", "twitter16", "source_tweets.txt")
    path_labels = os.path.join("data", "twitter16", "label.txt")
    path_tree_files = os.path.join("data", "twitter16", "tree")
    
    # Leggi i dati (puoi implementare funzioni di lettura separate se necessario)
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
    
    # Ottieni il driver di Neo4j e importa i dati
    driver = get_neo4j_driver()
    import_data_to_neo4j(driver, tweets, path_tree_files)
    driver.close()

if __name__ == '__main__':
    main()
