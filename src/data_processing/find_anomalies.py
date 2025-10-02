import os
import re
from collections import defaultdict

# Path to the directory containing tree files
data_dir = "rumor_detection_2017/twitter16/tree"
output_file = "anomalies.txt"

# Dictionary to keep track of tweet relationships
tweet_relations = defaultdict(list)

# Function to extract tuples from files
def parse_tree_file(filepath):
    """
    Parse a tree file and extract edges.
    Args:
        filepath (str): Path to the tree file.
    Returns:
        list of tuples: Each tuple contains (line, parent_tid, child_tid, parent_time
    """
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

# Function to find anomalies
def find_anomalies():
    """
    Find anomalies in tweet relationships.
    Returns:
        list of tuples: Each tuple contains (filename, line, reason, line_number)
    """
    anomalies = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            edges = parse_tree_file(filepath)
            
            for line, parent_tid, child_tid, parent_time, child_time, line_number in edges:
                # Save all connections
                tweet_relations[child_tid].append((parent_tid, parent_time))
                anomaly_type = 0

                # Inconsistency rules
                anomaly_descriptions = [
                    "Child before parent",
                    "Multiple inconsistent parents",
                    "Tweet is both parent and child of the same node"]
                
                conjuction = " & "

                if parent_time > child_time:
                    anomalies.append((filename, line, anomaly_descriptions[0], line_number))
                    anomaly_type = 1

                # Check for multiple parents for the same child tweet
                if len(set(t[0] for t in tweet_relations[child_tid])) > 1 and parent_tid != child_tid:
                    if anomaly_type == 1:  # If there's already an anomaly, replace it with the quadruple
                        anomalies.pop()  # Remove the last element from the list to put the quadruple and not print the same line twice
                        anomalies.append((filename, line, anomaly_descriptions[0] +conjuction+ anomaly_descriptions[1], line_number))
                        anomaly_type = 2
                    else:
                        anomalies.append((filename, line, anomaly_descriptions[1], line_number))
                        anomaly_type = 3

                # Check if the tweet is both parent and child
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

if __name__ == "__main__":
    # Find anomalies and save to file
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

    print(f"Results saved in {output_file}")
    # anomalies = find_anomalies()  # Questa riga è ridondante, puoi rimuoverla