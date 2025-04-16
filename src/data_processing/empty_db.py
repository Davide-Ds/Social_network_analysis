from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password!"

class Neo4jCleaner:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def drop_constraints(self):
        with self.driver.session() as session:
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                name = record["name"]
                print(f"🧨 Eliminazione vincolo: {name}")
                session.run(f"DROP CONSTRAINT {name}")

    def drop_indexes(self):
        with self.driver.session() as session:
            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                name = record["name"]
                print(f"🧨 Eliminazione indice: {name}")
                session.run(f"DROP INDEX {name}")

    def delete_all_data(self):
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
            print("🚀 Avvio cancellazione nodi con APOC...")
            result = session.run(apoc_query)
            summary = result.single()
            print(f"✅ Completato: {summary['total']} nodi in {summary['batches']} batch.")

    def full_clean(self):
        self.drop_constraints()
        self.drop_indexes()
        self.delete_all_data()


if __name__ == "__main__":
    cleaner = Neo4jCleaner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        cleaner.full_clean()
    finally:
        cleaner.close()
