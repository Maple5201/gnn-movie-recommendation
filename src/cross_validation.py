from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RANDOM_SEED = 42
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
EPOCHS = 100
LEARNING_RATE = 0.01
K_FOLDS = 5


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class GNNModel(torch.nn.Module):
    """A simple two-layer GraphSAGE model for rating prediction experiments."""

    def __init__(self, input_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM, output_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_ratings_1000.csv")


def build_id_maps(ratings: pd.DataFrame):
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())

    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_id_map = {movie_id: idx + len(user_ids) for idx, movie_id in enumerate(movie_ids)}

    return user_ids, movie_ids, user_id_map, movie_id_map


def build_graph_from_edges(ratings: pd.DataFrame, user_id_map: dict, movie_id_map: dict, num_nodes: int) -> Data:
    edges = []
    for user_id, movie_id in zip(ratings["userId"], ratings["movieId"]):
        if user_id in user_id_map and movie_id in movie_id_map:
            user_idx = user_id_map[user_id]
            movie_idx = movie_id_map[movie_id]
            edges.append([user_idx, movie_idx])
            edges.append([movie_idx, user_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.randn((num_nodes, EMBEDDING_DIM))

    return Data(x=node_features, edge_index=edge_index)


def predict_ratings(embeddings: torch.Tensor, ratings: pd.DataFrame, user_id_map: dict, movie_id_map: dict) -> torch.Tensor:
    predictions = []
    for user_id, movie_id in zip(ratings["userId"], ratings["movieId"]):
        if user_id in user_id_map and movie_id in movie_id_map:
            predictions.append(torch.dot(embeddings[user_id_map[user_id]], embeddings[movie_id_map[movie_id]]))

    if not predictions:
        return torch.empty(0)

    return torch.stack(predictions)


def main() -> None:
    ratings = load_ratings()
    user_ids, movie_ids, user_id_map, movie_id_map = build_id_maps(ratings)
    num_nodes = len(user_ids) + len(movie_ids)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(ratings), start=1):
        print(f"Fold {fold}/{K_FOLDS}")

        train_ratings = ratings.iloc[train_idx].reset_index(drop=True)
        val_ratings = ratings.iloc[val_idx].reset_index(drop=True)

        data_train = build_graph_from_edges(train_ratings, user_id_map, movie_id_map, num_nodes)

        model = GNNModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        y_train = torch.tensor(train_ratings["rating"].values, dtype=torch.float32)

        for _ in range(EPOCHS):
            model.train()
            optimizer.zero_grad()

            embeddings = model(data_train)
            y_pred_train = predict_ratings(embeddings, train_ratings, user_id_map, movie_id_map)

            loss = F.mse_loss(y_pred_train, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(data_train)
            y_pred_val = predict_ratings(embeddings, val_ratings, user_id_map, movie_id_map)
            y_val = torch.tensor(val_ratings["rating"].values[: len(y_pred_val)], dtype=torch.float32)

            rmse = np.sqrt(mean_squared_error(y_val.numpy(), y_pred_val.numpy()))
            rmse_scores.append(rmse)

        print(f"Validation RMSE for fold {fold}: {rmse:.4f}")

    print(f"Average RMSE across {K_FOLDS} folds: {np.mean(rmse_scores):.4f}")


if __name__ == "__main__":
    main()
