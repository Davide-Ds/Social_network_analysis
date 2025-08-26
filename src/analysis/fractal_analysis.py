import numpy as np
import random

def calculate_fractal_dimension(driver, max_box_size: int = 5, sample_size: int = 0):
    """
    Calcola la dimensione frattale del grafo:
    Args:
        driver: connessione Neo4j.
        max_box_size (int): distanza massima per il box-counting.
        sample_size (int): numero di tweet casuali da campionare, se 0 calcola su tutta la rete.

    Returns:
        dict: se sample_size > 0, restituisce un dizionario con tweet_id come chiave e dimensione frattale come valore;
        float: se sample_size = 0, stima della dimensione frattale dell'intera rete.
    """

    def fit_fractal_dimension(box_sizes, box_counts):
        """Helper per stimare la dimensione frattale da box-count e box-size."""
        if len(box_counts) < 2:
            return 0.0
        logs_eps = np.log(1 / np.array(box_sizes, dtype=float))
        logs_counts = np.log(np.array(box_counts, dtype=float))
        coeffs = np.polyfit(logs_eps, logs_counts, 1)
        return coeffs[0]

    # Caso 1: campionamento
    if sample_size > 0:
        print(f"Calcolo su un campione di {sample_size} tweet sorgenti...\n")

        with driver.session() as session:
            tweet_ids = [
                record["tweet_id"]
                for record in session.run("MATCH (t:Tweet) WHERE t.text IS NOT NULL RETURN t.tweet_id AS tweet_id")
            ]

        if len(tweet_ids) == 0:
            print("Nessun tweet sergente (cioè con attributo 'text') trovato.")
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
                        print(f"Box count 0 per box size {box_size}. Stop a {box_size-1}")
                        break
                    box_sizes.append(box_size)
                    box_counts.append(covered_total)

                print(f"Box size {box_sizes}\nBox count: {box_counts}")
                d = fit_fractal_dimension(box_sizes, box_counts)
                results[tid] = d
                print(f"Dimensione frattale = {d:.4f}\n")

    # Caso 2: intero grafo
    else:
        print("Calcolo sulla rete intera...\n")
        box_sizes, box_counts = [], []

        with driver.session() as session:
            for box_size in range(1, max_box_size + 1):
                query = f"""
                MATCH (n:Tweet)<-[*{box_size}]-(m:User)
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
        print(f"Dimensione frattale stimata sull'intera rete: {d:.4f}\n")