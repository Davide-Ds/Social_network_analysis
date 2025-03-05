import os
import re
from collections import defaultdict

# Percorso della cartella contenente i file di testo
data_dir = "rumor_detection_2017/twitter16/tree"
output_file = "anomalies.txt"

# Dizionario per tenere traccia delle relazioni
tweet_relations = defaultdict(list)

# Funzione per estrarre le tuple dai file
def parse_tree_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # Skip the first line ['ROOT', 'ROOT', '0.0']
        lines = f.readlines()
    
    edges = []
    pattern = re.compile(r"\['(.+?)', '(.+?)', '([\d.]+)'\]->\['(.+?)', '(.+?)', '([\d.]+)'\]")
    
    for line in lines:
        match = pattern.search(line)
        if match:
            parent_uid, parent_tid, parent_time, child_uid, child_tid, child_time = match.groups()
            parent_time, child_time = float(parent_time), float(child_time)
            edges.append((line.strip(), parent_tid, child_tid, parent_time, child_time))
    
    return edges

# Funzione per controllare le anomalie
def find_anomalies():
    anomalies = []
    i = 0
    for filename in os.listdir(data_dir):
      if i < 1:
        i+= 1
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            edges = parse_tree_file(filepath)
            
            for line, parent_tid, child_tid, parent_time, child_time in edges:
                # Salva tutte le connessioni
                tweet_relations[child_tid].append((parent_tid, parent_time))
                
                # Regole di incoerenza
                if parent_time > child_time:
                    anomalies.append((filename, line, "Child before parent"))
                if len(set(t[0] for t in tweet_relations[child_tid])) > 1:
                    print(set(t[0] for t in tweet_relations[child_tid]))
                    anomalies.append((filename, line, "Multiple inconsistent parents"))
                """if child_tid in tweet_relations and any(p[0] == child_tid for p in tweet_relations[child_tid]):
                    anomalies.append((filename, line, "Tweet is both parent and child"))"""
    
    return anomalies

# Trova anomalie e salva nel file
anomalies = find_anomalies()

with open(output_file, "w", encoding="utf-8") as f:
    current_file = None
    for filename, line, reason in anomalies:
        if filename != current_file:
            if current_file is not None:
                f.write("\n")
            f.write(f"{filename}:\n")
            current_file = filename
        f.write(f"{line}  ({reason})\n")

print(f"Risultati salvati in {output_file}")
