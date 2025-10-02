import sys  
from .data_processing.empty_db import Neo4jCleaner
from .analysis.link_prediction import build_link_prediction_dataset, generate_graphsage_embeddings
from .utils.neo4j_utils import create_indexes, get_neo4j_driver, compute_and_save_tweet_embeddings
from .data_processing.import_data import (
    load_tweets_and_labels,
    process_tree_files,
    import_tweet_nodes,
    import_retweets,
)
from .analysis.graph_analysis import *
from .clustering.community_detection import *  # Import the missing function
from .propagation_prediction.tweet_propagation_prediction import tweet_propagation_prediction_NN
import os
from .logs.log_writer import setup_logging
from .analysis.fractal_analysis import calculate_fractal_dimension
from .analysis.moebius_analysis import MoebiusAnalyzer
from .classification.tweet_classifier import train_and_evaluate  # ML classification function
from sklearn.model_selection import StratifiedKFold, cross_validate
# Initialize logging (default folder: utils)



def main(mode):
    """
    Main function to execute the social network analysis workflow.

    Depending on the mode, this function can:
        1: Load tweets and retweet data into Neo4j
        2: Perform graph analysis
        3: Both load and analyze
        4: Run ML text classification (tweet_classifier)

    Args:
        mode (int): Operation mode
    """
    setup_logging()
    # Paths to data
    path_source_tweets = os.path.join("data", "twitter16", "source_tweets.txt")
    path_labels = os.path.join("data", "twitter16", "label.txt")
    path_tree_files = os.path.join("data", "twitter16", "tree")

    driver = get_neo4j_driver()

    # ----------------------------
    # MODE 1: Load data
    # ----------------------------
    if mode == 1:
        # Clear Neo4j database if full
        cleaner = Neo4jCleaner(driver)
        try:
            cleaner.full_clean()
        finally:
            cleaner.close()
        print("Neo4j database cleared. Starting data import...")

        # Load tweets and labels
        print(f"Loading tweets and labels from {path_source_tweets} and {path_labels}...")
        tweets = load_tweets_and_labels(path_source_tweets, path_labels)
        print(f"{len(tweets)} tweets loaded.")

        # Process tree files to extract creators and RETWEET relationships
        print("Processing tree files to extract creators and RETWEET relationships...")
        retweet_relations = process_tree_files(path_tree_files, tweets)
        print(f"{len(retweet_relations)} RETWEET relationships extracted from tree files.")

        # Create indexes and import data
        print("Creating indexes...")
        create_indexes(driver)
        print("Importing tweet nodes (with creator) into Neo4j...")
        import_tweet_nodes(driver, tweets)
        print("Importing RETWEET relationships into Neo4j...")
        import_retweets(driver, retweet_relations)
        print("Data import completed.")

    # ----------------------------
    # MODE 2: Analyze graph
    # ----------------------------
    if mode == 2:
        # Basic graph analysis
        print("\nRetrieving basic statistics...")
        stats = basic_statistics(driver)
        print(f"Basic statistics: {stats}")

        print("\nRetrieving class statistics...")
        class_statistics = get_class_stats(driver)
        print(f"Class statistics:\n {class_statistics}\n")
            # Identify most retweeted users
        print("\nIdentifying most retweeted users...")
        most_retweeted = find_most_retweeted_users(driver)
        print(f"Users found: {most_retweeted}")

        # Identify users who retweet the most
        print("\nIdentifying frequent retweeters...")
        frequent_retweeters = find_frequent_retwetters(driver)
        print(f"Users found: {frequent_retweeters}")

        # Analyze diffusion for the most retweeted tweet
        most_retweeted_tweet = get_most_retweeted_tweet(driver)
        print(f"\nAnalyzing diffusion for tweet {most_retweeted_tweet}...")
        diffusion = analyze_diffusion_patterns(driver, most_retweeted_tweet)
        print(f"\nDiffusion for tweet {most_retweeted_tweet}:")
        for level in diffusion:
            print(f"Tree Level: {level['hop_level']}, Count users at level: {level['num_users_at_level']}, Users at level: {level['users_at_level']}\n")

        # Calculate fractal dimension of the whole network, set the parameter sample_size > 0 too use a sample of random tweets instead
        print(f"\nCalculating fractal dimension")
        calculate_fractal_dimension(driver, max_box_size= 5) 

        # Detect Möbius structures
        print("\nIdentifying Möbius structures in the social graph...")
        moebius = MoebiusAnalyzer(driver)
        try:
            moebius.show_and_visualize_structures(limit=5)
        finally:
            moebius.close()

        # Compute PageRank
        print("\nComputing PageRank...")
        create_User_gds_graph(driver)
        top_users = compute_pagerank(driver, 10)
        print("Top influential users (PageRank):")
        for user in top_users:
            print(f"User: {user['user']}, Score: {user['score']:.2f}")
            
        # Analyze top fake news creators
        print("\nAnalyzing top fake news creators...")
        top_fake_news_creators = get_top_fake_news_creators(driver, 10)
        print("Top influential fake news creators:")
        for creator in top_fake_news_creators:
            print(f"User: {creator['user_id']}, Total tweets: {creator['total_tweets']}, Fake News Count: {creator['num_fake_tweets']}, Fake tweets ids: {creator['fake_tweet_ids']}")
        
        # Link Prediction using GraphSAGE embeddings
        """
        Link Prediction Example using GraphSAGE Embeddings.

        This script demonstrates how to:
        1. Generate embeddings for User and Tweet nodes in Neo4j using GraphSAGE.
        2. Build a dataset of positive and negative User->Tweet pairs.
        3. Train a Random Forest classifier to predict RETWEET links.
        4. Evaluate the model using F1-score.

        Requirements:
        - neo4j
        - numpy
        - scikit-learn
        """
        print("Computing embeddings for tweets using all-MiniLM-L6-v2 model if not already present...") 
        if(not driver.session().run("MATCH (t:Tweet) WHERE exists(t.text_embedding) RETURN t LIMIT 1").single()):
            compute_and_save_tweet_embeddings(driver, model_name='all-MiniLM-L6-v2', text_property='text', embedding_property='text_embedding')
        print("Creating complete GDS graph with User and Tweet nodes...")
        create_complete_gds_graph(driver)
        
        # ---------------------------
        # Step 1: Generate embeddings
        # ---------------------------
        # Generate embeddings for User nodes
        print("\nGenerating GraphSAGE embeddings for Users...")
        if (driver.session().run("CALL gds.model.exists('UserSAGE') YIELD exists RETURN exists").single().value()):
            driver.session().run("CALL gds.model.drop('UserSAGE')"
        )        
        user_embeddings = generate_graphsage_embeddings(
            driver,       # Neo4j driver instance
            graph_name="fullGraph",
            model_name="UserSAGE",
            dim=128,
            node_label="User"
        )
            
        print("\nGenerating GraphSAGE embeddings for Tweets...")
        if (driver.session().run("CALL gds.model.exists('TweetSAGE') YIELD exists RETURN exists").single().value()):
            driver.session().run("CALL gds.model.drop('TweetSAGE')"
            )
        tweet_embeddings = generate_graphsage_embeddings(
            driver,
            graph_name="fullGraph",
            model_name="TweetSAGE",
            dim=128,
            node_label="Tweet"
        )
        
        # ---------------------------
        # Step 2: Build link prediction dataset
        # ---------------------------
        # Positive pairs: existing RETWEET relationships
        # Negative pairs: random User-Tweet pairs with no RETWEET
        print("Building link prediction dataset...")
        X, y = build_link_prediction_dataset(driver, user_embeddings, tweet_embeddings)

        # ---------------------------
        # Step 3: Train-test split
        # ---------------------------
        from sklearn.model_selection import train_test_split
        print("Splitting dataset into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------------------
        # Step 4: Train classifier
        # ---------------------------
        from sklearn.ensemble import RandomForestClassifier
        print("Training Random Forest classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
        print("Fitting model...")
        clf.fit(X_train, y_train)

        # ---------------------------
        # Step 5: Predict and evaluate
        # ---------------------------
        
        from sklearn.metrics import f1_score, accuracy_score

        # Evaluation on training set
        y_train_pred = clf.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Evaluation on test set
        y_test_pred = clf.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print("\nConfronto metriche:")
        print(f"Training set - F1-score: {train_f1:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Test set     - F1-score: {test_f1:.4f}, Accuracy: {test_acc:.4f}")
        print("Evaluating model...")
        y_pred = clf.predict(X_test)

        import pandas as pd
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

        # Compute metrics
        metrics = {
            "F1-score": [f1_score(y_test, y_pred)],
            "Accuracy": [accuracy_score(y_test, y_pred)],
            "Precision": [precision_score(y_test, y_pred)],
            "Recall": [recall_score(y_test, y_pred)]
        }

        df_metrics = pd.DataFrame(metrics)
        print("\nEvaluation Metrics:\n")
        print(df_metrics.to_string(index=False))

        # Confusion Matrix
        print("\nConfusion Matrix:\n")
        print(confusion_matrix(y_test, y_pred))

        # Cross-validation
        print("\nPerforming 10-fold cross-validation...")
        # Use StratifiedKFold to maintain class balance in each fold
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        scoring = ['accuracy', 'f1', 'precision', 'recall']
        results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)

        # Print scores for each fold and their averages
        print("\nCross-validation results:")
        for metric in scoring:
            scores = results[f'test_{metric}']
            print(f"{metric.capitalize()} per fold: {scores}")
            print(f"{metric.capitalize()} medio: {scores.mean():.4f} (+/- {scores.std():.4f})")


        
        
    # ----------------------------
    # MODE 3: ML text classification
    # ----------------------------
    if mode == 3:
        print("\nRunning ML text classification on tweets...")
        train_and_evaluate(path_source_tweets, path_labels)
    
    # ----------------------------
    # MODE 4: ML community detection
    # ----------------------------
    if mode == 4:
        print("\nRunning community detection using Leiden algorithm...")
        df_users = leiden_user_communities(driver)
        print(df_users.head())   # Print first few rows of the DataFrame
        print("Number of communities:", df_users['communityId'].nunique())

        # Write the DataFrame to a CSV file
        export_cluster_size_distribution(df_users, os.path.join("src","clustering","community_size_distribution.csv"))
        export_users_ordered_by_cluster(df_users, os.path.join("src","clustering","users_by_cluster.csv"))

        # Analyze and print details about the top communities
        df_anlysis= analyze_communities(driver, df_users, max_communities=5, output_csv_path= os.path.join("src","clustering","top_communities_analysis.csv"))
        print(df_anlysis.head())
    
    # ----------------------------
    # MODE 5: Tweet propagation prediction with Neural Networks
    # ----------------------------
    if mode == 5:
        print("\nRunning tweet propagation prediction with Neural Networks...")
        tweet = "Elon musk went to Mars on his tesla cybertruck"
        tweet_propagation_prediction_NN(driver, tweet)


    # ----------------------------
    # Close Neo4j connection
    # ----------------------------
    print("Closing Neo4j connection...")
    driver.close()

if __name__ == '__main__':
    """
    Program entry point.

    Prompts the user to select the mode:
        0: Exit
        1: Load data
        2: Graph analysis
        3: Both load and analyze
        4: ML text classification
    """
    print("Modes: 1 = Load data, 2 = Analysis, 3 = ML classification, 4 = Clustering, 5 = Tweet propagation prediction, 0 = Exit")
    while True:
        mode = input("Enter mode (1, 2, 3, 4, 5 or 0 to exit): ").strip()
        if mode == "0":
            print("Exiting program.")
            sys.exit(0)
        if mode in {"1", "2", "3", "4", "5"}:
            main(int(mode))
            break
        else:
            print("Invalid mode. Please try again.")