import os
from neo4j import GraphDatabase

def get_neo4j_driver():
    port = os.getenv("NEO4J_PORT", "7687")  # Default: 7687 se non impostata
    uri = f"bolt://localhost:{port}"
    username = os.getenv("NEO4J_USERNAME", "neo4j")  # Default: neo4j
    password = os.getenv("NEO4J_PASSWORD", "password!")  # Default: password!
    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver
