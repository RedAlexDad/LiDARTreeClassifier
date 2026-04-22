import itertools
from scripts.train import train_single
from scripts.load_data import load_dataset
from sklearn.model_selection import train_test_split

PARAM_GRID = {
    "epochs": [20, 30],
    "batch_size": [16, 32, 64],
    "lr": [0.01, 0.005, 0.001],
    "optimizer": ["Adam", "SGD"],
    "aug_factor": [4, 6, 8],
    "noise_std": [0.005, 0.01],
    "dropout": [0.3, 0.5],
}


def grid_search(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]

    best_val_acc = 0
    best_params = None
    best_run_id = None
    results_table = []

    total = 1
    for v in values:
        total *= len(v)
    print(f"Total combinations: {total}")

    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        params["num_points"] = 4096
        params["num_classes"] = 3

        print(f"[{i + 1}/{total}] {params}")
        model, metrics, run_id = train_single(params, X_train, y_train, X_test, y_test)

        row = {**params, **metrics, "run_id": run_id}
        results_table.append(row)

        if metrics["best_val_acc"] > best_val_acc:
            best_val_acc = metrics["best_val_acc"]
            best_params = params
            best_run_id = run_id

        print(f"  -> best_val_acc={metrics['best_val_acc']:.4f}")

    return best_params, best_val_acc, best_run_id, results_table


if __name__ == "__main__":
    X, y, classes = load_dataset()
    best_params, best_val_acc, best_run_id, results = grid_search(X, y)
    print(f"\nBest: {best_params}")
    print(f"Best val_acc: {best_val_acc}")
    print(f"Best run_id: {best_run_id}")
