# Social Network Analysis
Machine Learning course project, University of Bari "Aldo Moro".

## Overview
This repository implements a pipeline to import Twitter data into a Neo4j graph, perform graph analysis (diffusion, PageRank, fractal and Möbius analyses), compute node embeddings (GraphSAGE + text embeddings), run link-prediction experiments, train a text-based fake-news classifier, detect user communities (Leiden), and run tweet propagation prediction models.

## Authors
- Davide De Simone {d.desimone3@studenti.uniba.it}
- Bartolomeo Marcosano {b.marcosano@studenti.uniba.it}

## License
This project is licensed under the MIT License — see the `LICENSE` file.

## Key features
- Import tweets, labels and propagation trees into Neo4j.
- Graph analytics: basic stats, diffusion, PageRank, fractal dimension, Möbius structures.
- Node embeddings: GraphSAGE via Neo4j GDS and text embeddings via SentenceTransformers.
- Link prediction experiments using RandomForest over GraphSAGE embeddings.
- Tweet text classification: TF–IDF + Logistic Regression.
- Community detection (Leiden) and cluster exports.
- Tweet propagation prediction (neural network).

## Quick links
- Main runner: [src/main.py](src/main.py#L1-L200)
- Data import: [src/data_processing/import_data.py](src/data_processing/import_data.py#L1-L200)
- Neo4j helpers & embeddings: [src/utils/neo4j_utils.py](src/utils/neo4j_utils.py#L1-L200)
- Classifier: [src/classification/tweet_classifier.py](src/classification/tweet_classifier.py#L1-L200)
- Artifacts: `artifacts/`
- Packaging: [setup.py](setup.py)

## Requirements
- Python: > 3.10
- See [requirements.txt](requirements.txt) for full dependency list (Neo4j driver, sentence-transformers, scikit-learn, pandas, joblib, etc.).

## Installation
1. Create & activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix / macOS
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Install the package:
```bash
pip install -e .
```

## Configuration (Neo4j)
The project reads Neo4j connection settings from environment variables in `src/utils/neo4j_utils.py`.

Instructions:
- Set `NEO4J_PORT`, `NEO4J_USERNAME` and `NEO4J_PASSWORD` in your environment before running the code.
- Ensure your Neo4j instance has the APOC and GDS plugins enabled.

Note: example credentials are intentionally omitted from the README; configure them locally via environment variables.

## Data
- Datasets: `data/twitter16/`
	- `source_tweets.txt` — `tweet_id<TAB>text`
	- `label.txt` — `label:tweet_id`
	- `tree/` — propagation tree files used to derive RETWEET/QUOTE/INTERACTION relationships

## Usage
Run the interactive main script:
```bash
python src/main.py
```
Modes prompted by the script:
- `1` — Load data: clear Neo4j data base if full and import tweets + relationships.
- `2` — Graph analysis: stats, diffusion, fractal, Möbius, PageRank, fake-news creators.
- `3` — Link prediction: GraphSAGE embeddings, build dataset, train RandomForest, evaluate.
- `4` — Tweet text classification: TF–IDF + Logistic Regression (artifacts saved to `artifacts/`).
- `5` — Community detection (Leiden): exports CSVs to `src/clustering/`.
- `6` — Tweet propagation prediction: neural network example.
- `7` — Run all modes in sequence (1..6).
- `0` — Exit.

## Outputs & artifacts
- `artifacts/` — saved models and evaluation outputs (e.g. `model_logreg.joblib`, `tfidf_vectorizer.joblib`, `classification_report.txt`, `confusion_matrix.json`).
- Cluster CSVs: `src/clustering/community_size_distribution.csv`, `src/clustering/users_by_cluster.csv`, `src/clustering/top_communities_analysis.csv`.
- Logs: `src/logs/` (runtime logs created by `setup_logging()`).

## Implementation notes
- Import uses APOC to create dynamic relationship types and attaches `created_by` where appropriate.
- Text embeddings use `sentence-transformers` (default model typically used: `all-MiniLM-L6-v2`).
- Link prediction uses Neo4j GDS GraphSAGE models.
- Classification maps labels to binary `fake` vs `real` and uses a balanced Logistic Regression.

## How to reproduce experiments
1. Start Neo4j with APOC & GDS plugins enabled.
2. Place dataset files in `data/twitter16/`.
3. Set Neo4j environment variables and run `python src/main.py`, choose mode `1` to import, then `2`/`3`/`4` etc.

## Contributing
- Open issues for bugs/features.
- Fork, branch per feature, and open a PR.
- Keep code style consistent with existing repository.

## License & authors
- License: MIT — see the `LICENSE` file.
- Authors: Davide De Simone, Bartolomeo Marcosano

## References
- Project scripts and helper functions are in `src/` (see links above).
- Requirements: [requirements.txt](requirements.txt)
