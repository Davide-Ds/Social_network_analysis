.. SocNetAnalyzer documentation master file, created by
   sphinx-quickstart on Thu Oct  2 21:22:59 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SocNetAnalyzer documentation
============================

Welcome to the SocNetAnalyzer project documentation. This project provides a modular toolkit for analysing information propagation, community structure, and content characteristics in social networks built from Twitter data. It combines graph data science operations on Neo4j with data preprocessing utilities, clustering and community analysis, propagation prediction, and supervised classification of tweets.

Project overview
----------------
SocNetAnalyzer is organized into clear subpackages that separate responsibilities and allow reproducible experiments and evaluation:

- src.analysis
  - graph_analysis: extracts structural statistics, constructs GDS projections and computes centrality measures (PageRank) and diffusion patterns.
  - fractal_analysis: estimations of fractal dimension of diffusion trees and ensemble statistics on propagation structures.
  - link_prediction: utilities and experiments for predicting missing edges or potential retweets.
  - moebius_analysis: focused analyses combining user/tweet labels with graph structures to highlight propagation patterns.

- src.clustering
  - community_detection: identification and analysis of user communities using graph-clustering algorithms (Leiden via Neo4j GDS). Exports community distributions and supports keyword extraction and per-community summaries.

- src.classification
  - tweet_classifier: supervised models and feature pipelines for tweet-level classification (e.g. rumor / non-rumor). The implementation relies on scikit-learn pipelines and text feature extraction (TF-IDF, embeddings).

- src.propagation_prediction
  - tweet_propagation_prediction: models and evaluation code for predicting propagation metrics (reach, velocity, retweet counts) using features from both text and network structure.

- src.data_processing
  - import_data, find_anomalies, correct_anomalies, empty_db: robust ingestion of dataset trees, anomaly detection and correction utilities, and tools to reset or empty the Neo4j database.

- src.utils and src.logs
  - Utilities for Neo4j connection management, serialization helpers and a DualLogger utility to centralize logging output.

Machine learning and graph algorithms
-------------------------------------
The project integrates a blend of classical and graph-centric techniques to address social network analysis tasks:

- Graph Data Science (Neo4j GDS): graph projection, PageRank, Leiden community detection, and graph projections for algorithm execution.
- Community detection: Leiden algorithm for robust modularity-based clustering.
- Node and text embeddings: spaCy-based embeddings and custom procedures to attach text embeddings to Tweet nodes for downstream tasks.
- Supervised learning: scikit-learn classifiers (e.g. Logistic Regression, Random Forest, SVM) within standardized pipelines for feature extraction, selection and evaluation.
- Predictive modeling: regression and classification models for propagation metrics, leveraging both node-level graph features and textual representations.
- Evaluation: cross-validation, precision/recall/F1 for classification tasks and RMSE/MAE for regression-style propagation predictions.

Usage notes and reproducibility
-------------------------------
- The code is structured to run with a local Neo4j instance and a Python virtual environment. See the README for environment setup, required models (e.g. spaCy), and example commands.
- Documentation pages include API references for each module and practical notebooks or scripts for reproducing the main experiments.
- Data exports (CSV) and intermediate artifacts are kept under module-specific folders; tests and example pipelines are included where applicable.

Navigation
----------
Use the "src" page to browse module-level API documentation and click into each submodule for detailed function-level documentation, usage examples and expected input/output formats.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   src