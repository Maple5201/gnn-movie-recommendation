from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    folds = [1, 2, 3, 4, 5]
    gnn_rmse = [1.51, 1.52, 1.50, 1.51, 1.51]
    cf_rmse = [1.76, 1.74, 1.75, 1.75, 1.74]

    plt.figure(figsize=(8, 6))
    plt.plot(folds, gnn_rmse, marker="o", label="GNN RMSE")
    plt.plot(folds, cf_rmse, marker="x", linestyle="--", label="Collaborative Filtering RMSE")

    plt.title("Comparison of GNN and Collaborative Filtering Models")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "gnn_cf_comparison_chart.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
