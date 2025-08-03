"""
## 3. Calcolo della Dimensione Frattale della Rete
**Obiettivo:**  
Misurare la dimensione frattale (ad es. con il metodo del box-counting) per valutare l’autosimilarità dell’albero dei retweet.

**Task:**
- Esportare la struttura del grafo e implementare un algoritmo di box-counting per calcolare la dimensione di Hausdorff o una stima della dimensione frattale.
- Confrontare il valore ottenuto con quello del triangolo di Sierpinski (~1.585) e analizzare le differenze.

**Metriche:**
- Valore della dimensione frattale.
- Variazioni in funzione di differenti tweet o cluster.
"""
import numpy as np
import networkx as nx
import networkx as nx
from utils.neo4j_utils import get_retweet_subgraph

def calculate_fractal_dimension_from_neo4j(driver, tweet_id: str, min_box_size: int = 1, max_box_size: int = 15) -> float:
    """
    Calcola una stima della dimensione frattale della struttura di retweet di un tweet
    usando solo query Cypher su Neo4j (senza NetworkX).

    Args:
        driver: connessione Neo4j.
        tweet_id (str): ID del tweet sorgente.
        min_box_size (int): distanza minima.
        max_box_size (int): distanza massima.

    Returns:
        float: stima della dimensione frattale.
    """
    import numpy as np

    box_sizes = range(min_box_size, max_box_size + 1)
    box_counts = []

    with driver.session() as session:
        for box_size in box_sizes:
            query = f"""
            MATCH (start {{tweet_id: $tweet_id}})
            MATCH (start)<-[*1..{box_size}]-(n)
            RETURN count(DISTINCT n) AS box_count
            """
            result = session.run(query, tweet_id=tweet_id)
            single = result.single()
            if single is None or single["box_count"] is None:
                print(f"Warning: Nessun risultato per box_size={box_size}.")
                box_count = 0
            else:
                box_count = single["box_count"]
            box_counts.append(box_count)

    print(f"Box counts: {box_counts}")
    print(f"Box sizes: {list(box_sizes)}")

    # Controllo validità dei dati
    if len(box_counts) == 0:
        print("Warning: Nessun dato disponibile per il calcolo della dimensione frattale.")
        return 0.0
    if len(set(box_counts)) == 1 or min(box_counts) == 0:
        print("Box-counting non significativo: tutti i box_counts sono uguali o nulli.")
        return 0.0

    # Calcolo dei logaritmi, filtrando eventuali zeri
    valid_indices = [i for i, count in enumerate(box_counts) if count > 0]
    if len(valid_indices) < 2:
        print("Warning: Dati insufficienti per il fit logaritmico.")
        return 0.0

    logs_eps = np.log(1 / np.array([box_sizes[i] for i in valid_indices], dtype=float))
    logs_counts = np.log(np.array([box_counts[i] for i in valid_indices], dtype=float))

    if np.any(np.isnan(logs_eps)) or np.any(np.isnan(logs_counts)) or np.any(np.isinf(logs_eps)) or np.any(np.isinf(logs_counts)):
        print("Warning: Valori non validi nei logaritmi, impossibile stimare la dimensione frattale.")
        return 0.0

    coeffs = np.polyfit(logs_eps, logs_counts, 1)
    fractal_dimension = coeffs[0]
    if fractal_dimension < 0:
        print("Attenzione: dimensione frattale negativa, stima non significativa.")
    return