# GNN Movie Recommendation System

An experimental movie recommendation project using graph neural networks and contextual information. The project represents users and movies as graph nodes, with rating interactions as edges, and evaluates rating prediction performance using RMSE.

This project is based on my published paper:

**Movie Recommendation System Based on Graph Neural Network and Contextual Information**
DOI: `10.61173/7e0att59`

## Tech Stack

* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* PyTorch
* PyTorch Geometric
* GraphSAGE
* RMSE evaluation

## Project Structure

```text
gnn_movie_recommendation/
├── data/
│   ├── sample_links_1000.csv
│   ├── sample_movies_1000.csv
│   ├── sample_ratings_1000.csv
│   └── sample_tags_1000.csv
├── results/
├── src/
│   ├── data_exploration.py
│   ├── train_gnn_model.py
│   ├── cross_validation.py
│   └── plot_rmse_comparison.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Overview

The project explores a graph based approach to movie recommendation. Users and movies are represented as nodes, and user movie rating interactions are represented as edges.

A GraphSAGE based graph neural network model is used to learn user and movie representations. Rating prediction is then evaluated using RMSE.

The project also includes data exploration and visualisation scripts for understanding rating distribution, movie genres, user rating behaviour, and model performance.

## Main Features

* Loads and processes MovieLens style sample data
* Builds user movie interaction data for graph based recommendation
* Trains a GraphSAGE based GNN model for rating prediction
* Evaluates model performance using RMSE
* Applies five-fold cross-validation for experimental evaluation
* Compares GNN and collaborative filtering RMSE results
* Generates visualisations for rating distribution and RMSE comparison

## Scripts

| File                          | Description                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `src/data_exploration.py`     | Loads sample data and generates basic visualisations such as rating distribution, movie genre counts, and user rating counts |
| `src/train_gnn_model.py`      | Builds a user movie graph, trains a GraphSAGE based GNN model, and calculates RMSE                                           |
| `src/cross_validation.py`     | Runs five-fold cross-validation for experimental model evaluation                                                            |
| `src/plot_rmse_comparison.py` | Plots RMSE comparison between the GNN model and a collaborative filtering baseline                                           |

## Results

In the reported experiment, adding temporal contextual information improved RMSE from **1.51** to **1.45**.

The project also compares the GNN model against a collaborative filtering baseline using RMSE across five folds.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run data exploration:

```bash
python src/data_exploration.py
```

Generate the RMSE comparison chart:

```bash
python src/plot_rmse_comparison.py
```

Run the GNN training script:

```bash
python src/train_gnn_model.py
```

Run cross-validation:

```bash
python src/cross_validation.py
```

## Notes

This project was developed as an academic research project. It focuses on experimental model design, graph based recommendation, and evaluation rather than production deployment.

The sample data files are included for demonstration and reproducibility of the project structure.

## Limitations

* The project uses a sampled dataset rather than a full production-scale recommendation dataset.
* Node features are simplified for experimental modelling.
* The implementation focuses on academic evaluation and visualisation rather than deployment.
* Further work could improve feature engineering, model tuning, and comparison with additional recommendation baselines.
