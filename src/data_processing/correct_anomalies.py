import os
import re
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

def parse_tree_file(filepath):
    """
    Legge un file tree e restituisce una lista di edge.
    Ogni edge è una tupla:
    (line, parent_tid, child_tid, parent_time, child_time, line_number)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # Salta la prima riga (presumibilmente la radice)
        lines = f.readlines()
    
    edges = []
    pattern = re.compile(r"\['(.+?)', '(.+?)', '([\d.]+)'\]->\['(.+?)', '(.+?)', '([\d.]+)'\]")
    for line_number, line in enumerate(lines, start=2):
        match = pattern.search(line)
        if match:
            parent_uid, parent_tid, parent_time, child_uid, child_tid, child_time = match.groups()
            parent_time = float(parent_time)
            child_time = float(child_time)
            edges.append((line.strip(), parent_tid, child_tid, parent_time, child_time, line_number))
    return edges

def correct_anomalies_in_edges(edges):
    """
    Applica alcune regole di correzione:
      - Se il tempo del figlio è minore di quello del padre, forza il figlio a partire al medesimo tempo del padre.
      - Per un tweet figlio con più genitori, si mantiene la relazione con il delay minore.
      - Rimuove eventuali edge in cui il tweet appare sia come genitore che come figlio (cicli auto-riferiti).
    Restituisce una lista di edge corretti.
    """
    # Regola 1: Correggi i delay negativi (child_time < parent_time)
    corrected_edges = []
    for edge in edges:
        line, parent_tid, child_tid, parent_time, child_time, line_number = edge
        if child_time < parent_time:
            logging.info(f"Correzione: per edge a ln {line_number} il child_time ({child_time}) < parent_time ({parent_time}). Imposto child_time = {parent_time}.")
            child_time = parent_time
        corrected_edges.append((line, parent_tid, child_tid, parent_time, child_time, line_number))
    
    # Regola 2: Per tweet figlio con più genitori, mantieni quello con il delay minore
    grouped = defaultdict(list)
    for edge in corrected_edges:
        _, parent_tid, child_tid, parent_time, child_time, line_number = edge
        grouped[child_tid].append(edge)
    
    filtered_edges = []
    for child_tid, edge_list in grouped.items():
        # Se più edge per lo stesso child_tid, sceglie quello con il child_time minore
        best_edge = min(edge_list, key=lambda x: x[4])
        filtered_edges.append(best_edge)
    
    # Regola 3: Rimuovi gli edge in cui il tweet è sia padre che figlio (cicli)
    final_edges = [edge for edge in filtered_edges if edge[1] != edge[2]]
    
    return final_edges

def process_all_tree_files(data_dir):
    """
    Processa tutti i file tree nella cartella data_dir,
    applicando il parsing e la correzione delle anomalie.
    Restituisce un dizionario:
       chiave = nome_file, valore = lista di edge corretti.
    """
    corrected_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            edges = parse_tree_file(filepath)
            corrected_edges = correct_anomalies_in_edges(edges)
            corrected_data[filename] = corrected_edges
    return corrected_data

def import_corrected_retweets(driver, corrected_tree_data, retweet_limit=50000000):
    with driver.session() as session:
        for filename, edges in corrected_tree_data.items():
            # Recupera l'ID del tweet dal nome del file (ad es., "498430783699554305.txt" -> "498430783699554305")
            tweet_id = filename.replace(".txt", "")
            for edge in edges:
                # edge: (line, parent_tid, child_tid, parent_time, child_time, line_number)
                _, parent_tid, child_tid, _, child_time, _ = edge
                # Verifica che il tweet originale esista, se necessario
                # Crea il nodo per l'utente che retweetta
                session.run(
                    "MERGE (u:User {id: $user_id})",
                    user_id=child_tid  # supponiamo che child_tid rappresenti l'ID dell'utente che retweetta
                )
                # Crea la relazione RETWEETS con il delay corretto
                session.run(
                    "MATCH (u:User {id: $user_id}) "
                    "MATCH (t:Tweet {id: $tweet_id}) "
                    "MERGE (u)-[:RETWEETS {time: $retweet_time}]->(t)",
                    user_id=child_tid, tweet_id=parent_tid, retweet_time=str(child_time)
                )