import numpy as np
import random

def calculate_fractal_dimension(driver, max_box_size: int = 5, sample_size: int = 0):
    """
    Calculates the fractal dimension of the graph:
    Args:
        driver: Neo4j connection.
        max_box_size (int): maximum distance for box-counting.
        sample_size (int): number of random tweets to sample, if 0 computes on the whole network.

    Returns:
        dict: if sample_size > 0, returns a dictionary with tweet_id as key and fractal dimension as value;
        float: if sample_size = 0, estimate of the fractal dimension of the entire network.
    """

    def fit_fractal_dimension(box_sizes, box_counts):
        """Helper to estimate the fractal dimension from box-count and box-size."""
        if len(box_counts) < 2:
            return 0.0
        logs_eps = np.log(1 / np.array(box_sizes, dtype=float))
        logs_counts = np.log(np.array(box_counts, dtype=float))
        coeffs = np.polyfit(logs_eps, logs_counts, 1)
        return coeffs[0]

    # Case 1: sampling
    if sample_size > 0:
        print(f"Calculating on a sample of {sample_size} source tweets...\n")

        with driver.session() as session:
            tweet_ids = [
                record["tweet_id"]
                for record in session.run("MATCH (t:Tweet) WHERE t.text IS NOT NULL RETURN t.tweet_id AS tweet_id")
            ]

        if len(tweet_ids) == 0:
            print("No source tweets (i.e., with attribute 'text') found.")
            return 0.0

        sampled_ids = random.sample(tweet_ids, min(sample_size, len(tweet_ids)))
        results = {}

        with driver.session() as session:
            for tid in sampled_ids:
                box_sizes, box_counts = [], []
                print(f"Tweet {tid}:")
                for box_size in range(1, max_box_size + 1):
                    query = f"""
                    MATCH (start:Tweet {{tweet_id: $tid}})
                    MATCH (start)<-[*{box_size}]-(m:User)
                    RETURN count(DISTINCT m) AS covered
                    """
                    result = session.run(query, tid=tid).single()
                    covered_total = result["covered"]
                    if covered_total == 0:
                        print(f"Box count 0 for box size {box_size}. Stopping at {box_size-1}")
                        break
                    box_sizes.append(box_size)
                    box_counts.append(covered_total)

                print(f"Box size {box_sizes}\nBox count: {box_counts}")
                d = fit_fractal_dimension(box_sizes, box_counts)
                results[tid] = d
                print(f"Fractal dimension = {d:.4f}\n")

    # Case 2: entire graph
    else:
        print("Calculating on the entire network...\n")
        box_sizes, box_counts = [], []

        with driver.session() as session:
            for box_size in range(1, max_box_size + 1):    
                #users relates to tweets(create, interaction, quote, retweet) and to other users(retweeted_from)
                query = f"""
                MATCH (n)<-[*{box_size}]-(m:User)
                RETURN count(DISTINCT m) AS covered
                """
                result = session.run(query).single()
                covered_total = result["covered"]
                if covered_total == 0:
                    break
                box_sizes.append(box_size)
                box_counts.append(covered_total)

        d = fit_fractal_dimension(box_sizes, box_counts)
        print(f"Box size {box_sizes}\nBox count: {box_counts}")
        print(f"Estimated fractal dimension on the entire network: {d:.4f}\n")