import os
import re
import logging
from collections import defaultdict

#logging.basicConfig(level=logging.INFO)

def parse_tree_file(filepath):
    """
    Reads a tree file and returns a list of edges.
    
    Each edge is a tuple: 
    (line, parent_tid, child_tid, parent_time, child_time, line_number)
    
    Args:
        filepath (str): Path to the tree file.
        
    Returns:
        list of tuples: Each tuple contains (line, parent_tid, child_tid, parent_time
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
    Applies correction rules:
      - If the child's time is less than the parent's time, set the child's time equal to the parent's time.
      - For a child tweet with multiple parents, keep the relation with the smallest delay.
      - Removes edges where the tweet appears as both parent and child (self-referential cycles).
    
    Returns a list of corrected edges.
    
    Args:
        edges (list of tuples): List of edges as returned by parse_tree_file.
        
    Returns:
        list of tuples: Corrected list of edges.
    """
    # Rule 1: Fix negative delays (child_time < parent_time)
    corrected_edges = []
    for edge in edges:
        line, parent_tid, child_tid, parent_time, child_time, line_number = edge
        if child_time < parent_time:
            logging.info(f"Correction: per edge in ln {line_number} child_time ({child_time}) < parent_time ({parent_time}). Set child_time = {parent_time}.")
            child_time = parent_time
        corrected_edges.append((line, parent_tid, child_tid, parent_time, child_time, line_number))

    # Rule 2: For child tweet with multiple parents, keep the one with the smallest delay
    grouped = defaultdict(list)
    for edge in corrected_edges:
        _, parent_tid, child_tid, parent_time, child_time, line_number = edge
        grouped[child_tid].append(edge)
    
    filtered_edges = []
    for child_tid, edge_list in grouped.items():
        # If multiple edges for the same child_tid, choose the one with the smallest child_time
        best_edge = min(edge_list, key=lambda x: x[4])
        filtered_edges.append(best_edge)

    # Rule 3: Remove edges where the tweet is both parent and child (cycles)
    final_edges = [edge for edge in filtered_edges if edge[1] != edge[2]]
    
    return final_edges

def process_all_tree_files(data_dir):
    """
    Process all tree files in the data_dir folder,
    applying parsing and anomaly correction.
    
    Args:
        data_dir (str): Path to the directory containing tree files.
    
    Returns:
        dict: Mapping from filename to list of corrected edges.
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
    """
    Imports corrected retweet relationships into the Neo4j database.
    
    Args:
        driver: Neo4j driver.
        corrected_tree_data: dict mapping filename to list of corrected edges.
        retweet_limit: maximum number of retweets to import.
    
    Returns: 
        None
    """
    with driver.session() as session:
        for filename, edges in corrected_tree_data.items():
            # Retrieve the tweet ID from the filename (e.g., "498430783699554305.txt" -> "498430783699554305")
            tweet_id = filename.replace(".txt", "")
            for edge in edges:
                # edge: (line, parent_tid, child_tid, parent_time, child_time, line_number)
                _, parent_tid, child_tid, _, child_time, _ = edge
                # Check if the original tweet exists, if necessary
                # Create the node for the user who retweets
                session.run(
                    "MERGE (u:User {id: $user_id})",
                    user_id=child_tid  # assume child_tid represents the ID of the user who retweets
                )
                # Create the RETWEETS relationship with the corrected delay
                session.run(
                    "MATCH (u:User {id: $user_id}) "
                    "MATCH (t:Tweet {id: $tweet_id}) "
                    "MERGE (u)-[:RETWEETS {time: $retweet_time}]->(t)",
                    user_id=child_tid, tweet_id=parent_tid, retweet_time=str(child_time)
                )