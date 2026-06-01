from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
EPOCHS = 100
LEARNING_RATE = 0.01


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class GNNModel(torch.nn.Module):
    """A simple two-layer GraphSAGE model for learning node embeddings."""

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


def build_graph(ratings: pd.DataFrame):
    """Build a bipartite user-movie graph from rating interactions."""
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())

    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_id_map = {movie_id: idx + len(user_ids) for idx, movie_id in enumerate(movie_ids)}

    edges = []
    for user_id, movie_id in zip(ratings["userId"], ratings["movieId"]):
        user_idx = user_id_map[user_id]
        movie_idx = movie_id_map[movie_id]
        edges.append([user_idx, movie_idx])
        edges.append([movie_idx, user_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(user_ids) + len(movie_ids)
    node_features = torch.randn((num_nodes, EMBEDDING_DIM))

    data = Data(x=node_features, edge_index=edge_index)
    return data, user_id_map, movie_id_map


def predict_ratings(embeddings: torch.Tensor, ratings: pd.DataFrame, user_id_map: dict, movie_id_map: dict) -> torch.Tensor:
    predictions = [
        torch.dot(embeddings[user_id_map[user_id]], embeddings[movie_id_map[movie_id]])
        for user_id, movie_id in zip(ratings["userId"], ratings["movieId"])
    ]
    return torch.stack(predictions)


def plot_training_loss(losses: list[float]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss.png", bbox_inches="tight")
    plt.show()


def main() -> None:
    ratings = load_ratings()
    data, user_id_map, movie_id_map = build_graph(ratings)

    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    y_true = torch.tensor(ratings["rating"].values, dtype=torch.float32)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data)
        y_pred = predict_ratings(embeddings, ratings, user_id_map, movie_id_map)

        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(data)
        y_pred = predict_ratings(embeddings, ratings, user_id_map, movie_id_map)
        rmse = np.sqrt(mean_squared_error(y_true.numpy(), y_pred.numpy()))

    print(f"Training RMSE: {rmse:.4f}")
    plot_training_loss(losses)


if __name__ == "__main__":
    main()
