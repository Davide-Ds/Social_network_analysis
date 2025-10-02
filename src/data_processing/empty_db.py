from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password!"

class Neo4jCleaner:
    """
    A utility class for cleaning a Neo4j database. This includes:
    - Dropping all constraints.
    - Dropping all indexes.
    - Deleting all nodes and relationships using APOC procedures.
    """

    def __init__(self, driver):
        """
        Initialize the Neo4jCleaner with an existing Neo4j driver instance.
        :param driver: An existing Neo4j driver instance.
        """
        self.driver = driver

    @classmethod
    def from_credentials(cls, uri, user, password):
        """
        Alternative constructor to initialize the Neo4jCleaner with URI, username, and password.
        Args:
            uri (str): The URI of the Neo4j database.
            user (str): The username for authentication.
            password (str): The password for authentication.
        Returns: 
            Neo4jCleaner instance.
        """
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls(driver)

    def close(self):
        """
        Close the connection to the Neo4j database.
        """
        self.driver.close()

    def drop_constraints(self):
        """
        Drop all constraints in the Neo4j database.
        """
        with self.driver.session() as session:
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                name = record["name"]
                print(f" Eliminazione vincolo: {name}")
                session.run(f"DROP CONSTRAINT {name}")

    def drop_indexes(self):
        """
        Drop all indexes in the Neo4j database.
        """
        with self.driver.session() as session:
            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                name = record["name"]
                print(f" Eliminazione indice: {name}")
                session.run(f"DROP INDEX {name}")

    def delete_all_data(self):
        """
        Delete all nodes and relationships in the Neo4j database using APOC procedures.
        """
        apoc_query = """
        CALL apoc.periodic.iterate(
          "MATCH (n) RETURN n",
          "DETACH DELETE n",
          {batchSize: 100, parallel: false}
        )
        YIELD batches, total
        RETURN batches, total
        """
        with self.driver.session() as session:
            print(" Avvio cancellazione nodi con APOC...")
            result = session.run(apoc_query)
            summary = result.single()
            print(f"Completato: {summary['total']} nodi in {summary['batches']} batch.")

    def is_database_empty(self):
        """
        Check if the database is already empty.
        Returns:
            bool: True if the database is empty, False otherwise.
        """
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n) AS count")
            count = result.single()["count"]
            return count == 0

    def full_clean(self):
        """
        Perform a full cleanup of the Neo4j database by:
        1. Dropping all constraints.
        2. Dropping all indexes.
        3. Deleting all nodes and relationships.
        """
        if self.is_database_empty():
            print("The database is already empty. No cleanup is required.")
            return
        print("Database non vuoto, Inizio pulizia del database Neo4j...")
        self.drop_constraints()
        self.drop_indexes()
        self.delete_all_data()


if __name__ == "__main__":
    # Example usage with URI, username, and password
    cleaner = Neo4jCleaner.from_credentials(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        cleaner.full_clean()
    finally:
        cleaner.close()
