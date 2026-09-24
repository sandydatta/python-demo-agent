"""
Demonstration of core ML & Data Science libraries:
- numpy
- pandas
- matplotlib
- scikit-learn
- torch (PyTorch)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def demo_numpy() -> Dict[str, Any]:
    logger.info("Running NumPy demo...")
    arr = np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
    mean_val = float(np.mean(arr))
    dot_product = float(np.dot([1, 2], [3, 4]))
    return {
        "array_shape": arr.shape,
        "array_mean": mean_val,
        "dot_product": dot_product,
    }


def demo_pandas() -> Dict[str, Any]:
    logger.info("Running Pandas demo...")
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 40],
        "score": [88.5, 92.0, 79.5, 95.0],
    }
    df = pd.DataFrame(data)
    high_scorers = df[df["score"] > 90]
    avg_score = float(df["score"].mean())
    return {
        "total_rows": len(df),
        "high_scorers": high_scorers["name"].tolist(),
        "average_score": avg_score,
    }


def demo_matplotlib(output_dir: Path) -> str:
    logger.info("Running Matplotlib demo...")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "sample_plot.png"

    x = np.linspace(0, 10, 50)
    y = np.sin(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="sin(x)", color="teal", linewidth=2)
    plt.title("Sample Matplotlib Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    return str(plot_path)


def demo_scikit_learn() -> Dict[str, Any]:
    logger.info("Running Scikit-Learn demo...")
    # Synthetic dataset: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.float32)
    y = np.array([3, 5, 7, 9, 11, 13, 15, 17], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test).tolist()
    coef = float(model.coef_[0])
    intercept = float(model.intercept_)

    return {
        "model_coef": round(coef, 2),
        "model_intercept": round(intercept, 2),
        "predictions": [round(p, 2) for p in preds],
    }


def demo_pytorch() -> Dict[str, Any]:
    logger.info("Running PyTorch demo...")
    # Simple Linear Module forward pass
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleLinear()
    sample_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = model(sample_input)

    return {
        "input_shape": list(sample_input.shape),
        "output_shape": list(output.shape),
        "device": str(next(model.parameters()).device),
    }


def run_all_ml_libraries_demo(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = Path("./demo_output")

    np_res = demo_numpy()
    pd_res = demo_pandas()
    plot_path = demo_matplotlib(output_dir)
    sklearn_res = demo_scikit_learn()
    torch_res = demo_pytorch()

    return {
        "numpy": np_res,
        "pandas": pd_res,
        "matplotlib_saved_plot": plot_path,
        "scikit_learn": sklearn_res,
        "pytorch": torch_res,
    }


if __name__ == "__main__":
    import json
    res = run_all_ml_libraries_demo()
    print(json.dumps(res, indent=2))
