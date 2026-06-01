from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load the sampled MovieLens-style CSV files."""
    ratings = pd.read_csv(DATA_DIR / "sample_ratings_1000.csv")
    movies = pd.read_csv(DATA_DIR / "sample_movies_1000.csv")
    tags = pd.read_csv(DATA_DIR / "sample_tags_1000.csv")
    links = pd.read_csv(DATA_DIR / "sample_links_1000.csv")
    return ratings, movies, tags, links


def plot_rating_distribution(ratings: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(ratings["rating"], bins=10, edgecolor="black")
    plt.title("User Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rating_distribution.png", bbox_inches="tight")
    plt.show()


def plot_movies_per_genre(movies: pd.DataFrame) -> None:
    genre_series = movies["genres"].str.split("|").explode()

    plt.figure(figsize=(10, 5))
    genre_series.value_counts().plot(kind="bar")
    plt.title("Number of Movies per Genre")
    plt.xlabel("Genre")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "movies_per_genre.png", bbox_inches="tight")
    plt.show()


def plot_user_rating_count(ratings: pd.DataFrame) -> None:
    user_rating_count = ratings.groupby("userId").size()

    plt.figure(figsize=(6, 5))
    plt.boxplot(user_rating_count)
    plt.title("Boxplot of Number of Ratings per User")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "user_rating_count_boxplot.png", bbox_inches="tight")
    plt.show()


def main() -> None:
    ratings, movies, _, _ = load_data()
    plot_rating_distribution(ratings)
    plot_movies_per_genre(movies)
    plot_user_rating_count(ratings)


if __name__ == "__main__":
    main()
