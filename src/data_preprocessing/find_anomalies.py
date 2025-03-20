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
    
    for line_number, line in enumerate(lines, start=2):  # Start from 2 to account for the skipped line
        match = pattern.search(line)
        if match:
            parent_uid, parent_tid, parent_time, child_uid, child_tid, child_time = match.groups()
            parent_time, child_time = float(parent_time), float(child_time)
            edges.append((line.strip(), parent_tid, child_tid, parent_time, child_time, line_number))
    
    return edges

# Funzione per controllare le anomalie
def find_anomalies():
    anomalies = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            edges = parse_tree_file(filepath)
            
            for line, parent_tid, child_tid, parent_time, child_time, line_number in edges:
                # Salva tutte le connessioni
                tweet_relations[child_tid].append((parent_tid, parent_time))
                anomaly_type = 0

                # Regole di incoerenza
                anomaly_descriptions = [
                    "Child before parent",
                    "Multiple inconsistent parents",
                    "Tweet is both parent and child of the same node"]
                
                conjuction = " & "

                if parent_time > child_time:
                    anomalies.append((filename, line, anomaly_descriptions[0], line_number))
                    anomaly_type = 1

                # Controlla se ci sono più genitori per lo stesso tweet figlio
                if len(set(t[0] for t in tweet_relations[child_tid])) > 1 and parent_tid != child_tid:
                    if anomaly_type == 1:  # se c'è già un'anomalia, la sostituisce con la quadrupla
                        anomalies.pop()  # Rimuove l'ultimo elemento della lista per mettere la quadrupla e non stampare 2 volte la stessa linea
                        anomalies.append((filename, line, anomaly_descriptions[0] +conjuction+ anomaly_descriptions[1], line_number))
                        anomaly_type = 2
                    else:
                        anomalies.append((filename, line, anomaly_descriptions[1], line_number))
                        anomaly_type = 3
                
                # Controlla se il tweet è sia genitore che figlio
                if child_tid in tweet_relations and any(p[0] == child_tid for p in tweet_relations[child_tid]) and parent_tid != child_tid:
                    if anomaly_type == 1:
                        anomalies.pop()
                        anomalies.append((filename, line, anomaly_descriptions[0] +conjuction+ anomaly_descriptions[2], line_number))
                    elif anomaly_type == 2:
                        anomalies.pop()
                        anomalies.append((filename, line, anomaly_descriptions[0] +conjuction+ anomaly_descriptions[1] +conjuction+ anomaly_descriptions[2], line_number))
                    elif anomaly_type == 3:
                        anomalies.pop()
                        anomalies.append((filename, line, anomaly_descriptions[1] +conjuction+ anomaly_descriptions[2], line_number))
                    else:
                        anomalies.append((filename, line, anomaly_descriptions[2], line_number))

    return anomalies

# Trova anomalie e salva nel file
anomalies = find_anomalies()

with open(output_file, "w", encoding="utf-8") as f:
    current_file = None
    f.write("Parent node -> Child node\n" + "Each node is given as a tuple: ['uid', 'tweet ID', 'post time delay (in minutes)']->['uid', 'tweet ID', 'post time delay (in minutes)']\n")
    for filename, line, reason, line_number in anomalies:
        if filename != current_file:
            if current_file is not None:
                f.write("\n")
            f.write(f"\n{filename}:\n")
            current_file = filename
        f.write(f"{line}  ({reason}) [ln {line_number}]\n")

print(f"Risultati salvati in {output_file}")
