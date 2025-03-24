from src.utils.neo4j_utils import get_neo4j_driver
from src.data_processing.import_data import import_tweet_nodes, load_tweets_and_labels
from src.data_processing.correct_anomalies import process_all_tree_files, import_corrected_retweets
from src.analysis.graph_analysis import basic_statistics, find_influencers, analyze_diffusion_patterns
import os
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Percorsi dei dati
    path_source_tweets = os.path.join("data", "twitter16", "source_tweets.txt")
    path_labels = os.path.join("data", "twitter16", "label.txt")
    path_tree_files = os.path.join("data", "twitter16", "tree")
    
    # Leggi i dati
    tweets = load_tweets_and_labels(path_source_tweets, path_labels)
    
    # Ottieni il driver di Neo4j e importa i dati
    driver = get_neo4j_driver()

    import_tweet_nodes(driver, tweets)
    
    corrected_tree_data = process_all_tree_files(path_tree_files)
    import_corrected_retweets(driver, corrected_tree_data)
    
    # Analisi del grafo
    logging.info("Recupero statistiche di base...")
    stats = basic_statistics(driver)
    logging.info(stats)
    
    logging.info("Identificazione degli influencer...")
    influencers = find_influencers(driver)
    logging.info(influencers)
    
    tweet_id = "656955120626880512"  # Sostituire con un ID di tweet interessante
    logging.info(f"Analisi della diffusione per il tweet {tweet_id}...")
    diffusion = analyze_diffusion_patterns(driver, tweet_id)
    logging.info(diffusion)
    
    driver.close()

if __name__ == '__main__':
    main()
