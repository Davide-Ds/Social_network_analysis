import sys  
from data_processing.empty_db import Neo4jCleaner
from utils.neo4j_utils import create_indexes, get_neo4j_driver
from data_processing.import_data import (
    load_tweets_and_labels,
    process_tree_files,
    import_tweet_nodes,
    import_retweets,
)
from analysis.graph_analysis import *
import os
from logs.log_writer import setup_logging
from analysis.fractal_analysis import calculate_fractal_dimension
from analysis.moebius_analysis import MoebiusAnalyzer

setup_logging()  # Usa la cartella "utils" di default


def main(mode):
    # Percorsi dei dati
    path_source_tweets = os.path.join("data", "twitter16", "source_tweets.txt")
    path_labels = os.path.join("data", "twitter16", "label.txt")
    path_tree_files = os.path.join("data", "twitter16", "tree")

    driver = get_neo4j_driver()

    if mode in [1, 3]:  # Caricamento dati
        # Svuota il database Neo4j se pieno
        cleaner = Neo4jCleaner(driver)
        try:
            cleaner.full_clean()
        finally:
            cleaner.close()
        print("Database Neo4j vuoto/svuotato. Inizio importazione dati...")
        
        # 1. Caricamento dei tweet e delle etichette
        print(f"Caricamento dei tweet e delle etichette da {path_source_tweets} e {path_labels}...")
        tweets = load_tweets_and_labels(path_source_tweets, path_labels)
        print(f"{len(tweets)} tweet caricati.")
        
        # 2. Elaborazione dei file tree
        print("Elaborazione dei file tree per estrarre creatori e relazioni RETWEET...")
        retweet_relations = process_tree_files(path_tree_files, tweets)
        print(f"{len(retweet_relations)} relazioni RETWEET elaborate dai file tree.")
        
        # 3. Creazione degli indici e importazione dati
        print("Creazione degli indici...")
        create_indexes(driver)
        print("Importazione dei nodi dei tweet (con creatore) in Neo4j...")
        import_tweet_nodes(driver, tweets)
        print("Importazione delle relazioni RETWEET in Neo4j...")
        import_retweets(driver, retweet_relations)
        print("Importazione completata.")

    if mode in [2, 3]:  # Analisi
        
        # 4. Esecuzione di analisi sul grafo
        print("\nRecupero statistiche di base...")
        stats = basic_statistics(driver)
        print(f"Statistiche di base: {stats}")

        print("\nRecupero statistiche sulle classi...")
        class_statistics = get_class_stats(driver)
        print(f"Statistiche per classe:\n {class_statistics}\n")
        
        print("\nIdentificazione degli utenti più retweettati...")
        most_retweeted = find_most_retweeted_users(driver)
        print(f"Utenti trovati: {most_retweeted}")
        
        print("\nIdentificazione degli utenti che retweettano maggiormente...")
        frequent_retweeters = find_frequent_retwetters(driver)
        print(f"Utenti trovati: {frequent_retweeters}")
        
        most_retweeted_tweet = get_most_retweeted_tweet(driver)                                 
        print(f"\nAnalisi della diffusione per il tweet {most_retweeted_tweet}...")
        diffusion = analyze_diffusion_patterns(driver, most_retweeted_tweet)
        print(f"\nDiffusione per il tweet {most_retweeted_tweet}:")
        for level in diffusion:
            print(f"Tree Level: {level['hop_level']}, Count users at level: {level['num_users_at_level']}, Users at level: {level['users_at_level']}\n")  #paths_at_level: {level["paths_at_level"]}
     
        print(f"\nCalcolo della dimensione frattale")
        calculate_fractal_dimension(driver, max_box_size= 10)       
         
        print("\nIdentifying Möbius structures in the social graph...")
        moebius = MoebiusAnalyzer(driver)
        try:
            moebius.show_and_visualize_structures(limit=5)
        finally:
            moebius.close()
             
        # Calcolo del PageRank
        print("\nCalcolo del PageRank...")
        create_gds_graph(driver)
        top_users = compute_pagerank(driver, 10)
        print("Utenti più influenti:")
        for user in top_users:
            print(f"User: {user['user']}, Score: {user['score']:.2f}")
        
        #Trova i maggiori creatori di fake news
        print("\nAnalisi dei creatori di fake news...")
        top_fake_news_creators = get_top_fake_news_creators(driver, 10)
        print("Creatori di fake news più influenti:")
        for creator in top_fake_news_creators:
            print(f"User: {creator['user_id']}, Total tweets: {creator['total_tweets']}, Fake News Count: {creator['num_fake_tweets']}, Fake tweets ids: {creator['fake_tweet_ids']}")
           
    # Chiusura della connessione a Neo4j
    print("Chiusura della connessione a Neo4j...")
    driver.close()

if __name__ == '__main__':
    print("Modes: 1 = Caricamento dati, 2 = Analisi, 3 = Entrambi, 0 = Esci")
    while True:
        mode = input("Inserisci la modalità (1, 2, 3 o 0 per uscire): ").strip()
        if mode == "0":
            print("Uscita dal programma.")
            sys.exit(0)
        if mode in {"1", "2", "3"}:
            main(int(mode))
            break
        else:
            print("Modalità non valida. Riprova.")
