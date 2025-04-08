from utils.neo4j_utils import create_indexes, get_neo4j_driver
from data_processing.import_data import (
    load_tweets_and_labels,
    process_tree_files,
    import_tweet_nodes,
    import_retweets,
    get_most_retweeted_tweet
)
from analysis.graph_analysis import *
import os
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Percorsi dei dati
    path_source_tweets = os.path.join("data", "twitter16", "source_tweets.txt")
    path_labels = os.path.join("data", "twitter16", "label.txt")
    path_tree_files = os.path.join("data", "twitter16", "tree")
    
    # 1. Caricamento dei tweet e delle etichette
    logging.info(f"Caricamento dei tweet e delle etichette da {path_source_tweets} e {path_labels}...")
    tweets = load_tweets_and_labels(path_source_tweets, path_labels)
    logging.info(f"{len(tweets)} tweet caricati.")
    
    # 2. Elaborazione dei file tree:
    #    - Aggiorna il dizionario 'tweets' aggiungendo la proprietà 'created_by'
    #    - Raccoglie le relazioni RETWEET come lista di tuple (retweeter, tweet_id, creation_delay)
    logging.info("Elaborazione dei file tree per estrarre creatori e relazioni RETWEET...")
    retweet_relations = process_tree_files(path_tree_files, tweets)
    logging.info(f"{len(retweet_relations)} relazioni RETWEET elaborate dai file tree.")
    
    # 3. Connessione a Neo4j e creazione degli indici
    logging.info("Inizio connessione a Neo4j...")
    driver = get_neo4j_driver()
    logging.info("Creazione degli indici...")
    create_indexes(driver)
    
    # 4. Importazione dei nodi Tweet (con le proprietà aggiornate: text, tweet_label, created_by)
    logging.info("Importazione dei nodi dei tweet (con creatore) in Neo4j...")
    import_tweet_nodes(driver, tweets)
    
    # 5. Importazione delle relazioni RETWEET nel grafo
    logging.info("Importazione delle relazioni RETWEET in Neo4j...")
    import_retweets(driver, retweet_relations)
    
    # 6. Esecuzione di analisi sul grafo
    logging.info("Recupero statistiche di base...")
    stats = basic_statistics(driver)
    logging.info(f"Statistiche di base: {stats}")
    
    logging.info("Identificazione degli influencer...")
    influencers = find_influencers(driver)
    logging.info(f"Influencer trovati: {influencers}")
    
    most_retweeted_tweet = get_most_retweeted_tweet(driver)
    logging.info(f"Analisi della diffusione per il tweet {most_retweeted_tweet}...")
    diffusion = analyze_diffusion_patterns(driver, most_retweeted_tweet)
    logging.info(f"Diffusione per il tweet {most_retweeted_tweet}: {diffusion}")
    # 7. Chiusura della connessione a Neo4j
    
    # Calcola il PageRank e mostra i risultati
    print("Calcolo del PageRank...")
    top_users = compute_pagerank(driver, top_n=10)
    print("Utenti più influenti (PageRank):")
    for user in top_users:
        print(f"User: {user['user']}, Score: {user['score']:.4f}")       
    logging.info("Chiusura della connessione a Neo4j...")
    driver.close()

if __name__ == '__main__':
    main()
