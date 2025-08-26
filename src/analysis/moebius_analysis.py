from neo4j import GraphDatabase
from typing import List, Dict, Any
import urllib.parse
import webbrowser

class MoebiusAnalyzer:
    """
    A class to analyze Möbius-like structures in a social graph.\n
    A Möbius strip is a one-sided surface with no boundaries, created by taking a strip of paper, 
    giving it a half-twist, and joining the ends together. It is a well-known object in topology, 
    often used to illustrate concepts of non-orientability and continuity.
    
    The concept of a Möbius structure is used metaphorically to identify unique, 
    twisted relationship patterns within a social graph. These structures may represent complex, 
    recursive interactions—such as feedback loops or paradoxical roles—where connections seem to 
    "twist back" on themselves. The algorithm searches the graph database for such Möbius-like patterns, 
    highlights them, and visualizes them through Neo4j to support deeper analysis of social dynamics.    
    """

    def __init__(self, driver):
        self.driver = driver

    def close(self):
        pass

    def find_moebius_structures(self, limit: int = 100) -> List[Dict[str, Any]]:
        query = f"""
        MATCH 
          (u1:User)-[:CREATES]->(t1:Tweet)<-[:RETWEET]-(u2:User),
          (u2)-[:CREATES]->(t2:Tweet)<-[:RETWEET]-(u3:User),
          (u3)-[:CREATES]->(t3:Tweet)<-[:RETWEET]-(u1)
        RETURN 
          u1.user_id AS user1, t1.tweet_id AS tweet1,
          u2.user_id AS user2, t2.tweet_id AS tweet2,
          u3.user_id AS user3, t3.tweet_id AS tweet3
        LIMIT {limit}
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    @staticmethod
    def visualize_moebius_structure(
        moebius_structures: List[Dict[str, Any]], 
        index: int = 0, 
        neo4j_browser_url: str = "http://localhost:7474/browser/"
):
        if index < 0 or index >= len(moebius_structures):
            raise IndexError("Invalid index for Möbius structure list.")

        structure = moebius_structures[index]

        user1, tweet1 = structure["user1"], structure["tweet1"]
        user2, tweet2 = structure["user2"], structure["tweet2"]
        user3, tweet3 = structure["user3"], structure["tweet3"]

        cypher_query = f"""
        MATCH 
          (u1:User {{user_id: '{user1}'}})-[:CREATES]->(t1:Tweet {{tweet_id: '{tweet1}'}})<-[:RETWEET]-(u2:User {{user_id: '{user2}'}}),
          (u2)-[:CREATES]->(t2:Tweet {{tweet_id: '{tweet2}'}})<-[:RETWEET]-(u3:User {{user_id: '{user3}'}}),
          (u3)-[:CREATES]->(t3:Tweet {{tweet_id: '{tweet3}'}})<-[:RETWEET]-(u1)
        RETURN u1, t1, u2, t2, u3, t3
        """

        encoded_query = urllib.parse.quote(cypher_query)
        full_url = f"{neo4j_browser_url}?cmd=edit&arg={encoded_query}"

        print(f"[INFO] Opening Neo4j Browser for Möbius structure #{index}")
        print(f"[URL] {full_url}")
        webbrowser.open(full_url)

    
    def show_and_visualize_structures(self, limit=5):
        """
        Show and visualize Möbius structures in the social graph.
        All structures are visualized together in the Neo4j Browser.
        If more than one structure is found, they are displayed in a single view MAKING UNIONS of
        edges with same id.
        Args:
            limit (int): Maximum number of structures to visualize.
        
        Returns:
            None           
        """
        structures = self.find_moebius_structures(limit=limit)
        if structures:
            print(f"Found {len(structures)} Möbius structures. Generating a single visualization on Neo4j Browser...")
            match_clauses = []
            return_items = []
            for idx, structure in enumerate(structures):
                user1, tweet1 = structure["user1"], structure["tweet1"]
                user2, tweet2 = structure["user2"], structure["tweet2"]
                user3, tweet3 = structure["user3"], structure["tweet3"]
                match_clauses.append(
                    f"MATCH (u{idx}1:User {{user_id: '{user1}'}})-[:CREATES]->(t{idx}1:Tweet {{tweet_id: '{tweet1}'}})<-[:RETWEET]-(u{idx}2:User {{user_id: '{user2}'}})"
                    f"\nMATCH (u{idx}2)-[:CREATES]->(t{idx}2:Tweet {{tweet_id: '{tweet2}'}})<-[:RETWEET]-(u{idx}3:User {{user_id: '{user3}'}})"
                    f"\nMATCH (u{idx}3)-[:CREATES]->(t{idx}3:Tweet {{tweet_id: '{tweet3}'}})<-[:RETWEET]-(u{idx}1)"
                    )
                return_items.extend([
                    f"u{idx}1", f"t{idx}1", f"u{idx}2", f"t{idx}2", f"u{idx}3", f"t{idx}3"
                    ])
            cypher_query = f"{chr(10).join(match_clauses)}\nRETURN {', '.join(return_items)}"
            encoded_query = urllib.parse.quote(cypher_query)
            full_url = f"http://localhost:7474/browser/?cmd=edit&arg={encoded_query}"
            print(f"[INFO] Opening Neo4j Browser for all Möbius structures")
            print(f"[URL] {full_url}")
            webbrowser.open(full_url)
            print("Open this link in your browser to view all Möbius structures together.")
        else:
            print("No Möbius structures found.")