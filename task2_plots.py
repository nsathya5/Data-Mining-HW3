#!/usr/bin/env python3
"""Plotting functions for Task 2 recommender system experiments."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def plot_similarity_bar_charts(results: List) -> None:
    """Create 2 separate bar charts for similarity metrics (RMSE and MAE)."""
    ensure_results_dir()
    
    sims = ["cosine", "msd", "pearson"]
    x_pos = np.arange(len(sims))
    width = 0.35
    
    # Prepare data
    user_rmse, item_rmse, user_mae, item_mae = [], [], [], []
    
    for sim_name in sims:
        for r in results:
            if r.detail == f"sim={sim_name}":
                if r.model == "UserCF":
                    user_rmse.append(r.rmse)
                    user_mae.append(r.mae)
                elif r.model == "ItemCF":
                    item_rmse.append(r.rmse)
                    item_mae.append(r.mae)
    
    # Create RMSE bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos - width/2, user_rmse, width, label="UserCF", color="orange", alpha=0.8)
    ax.bar(x_pos + width/2, item_rmse, width, label="ItemCF", color="blue", alpha=0.8)
    ax.set_xlabel("Similarity Metric", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("RMSE by Similarity Metric", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sims)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    output_path = RESULTS_DIR / "task2_similarity_rmse.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Task2] Saved RMSE bar chart to {output_path}")
    
    # Create MAE bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos - width/2, user_mae, width, label="UserCF", color="orange", alpha=0.8)
    ax.bar(x_pos + width/2, item_mae, width, label="ItemCF", color="blue", alpha=0.8)
    ax.set_xlabel("Similarity Metric", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("MAE by Similarity Metric", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sims)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    output_path = RESULTS_DIR / "task2_similarity_mae.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Task2] Saved MAE bar chart to {output_path}")


def plot_neighbor_sweep(results: List, neighbors: List[int]) -> None:
    """Create 2 separate line charts for neighbor sweep (RMSE and MAE)."""
    ensure_results_dir()
    
    k_values = sorted(neighbors)
    user_rmse, item_rmse, user_mae, item_mae = [], [], [], []
    
    for k in k_values:
        for r in results:
            if r.detail == f"k={k}":
                if r.model == "UserCF":
                    user_rmse.append(r.rmse)
                    user_mae.append(r.mae)
                elif r.model == "ItemCF":
                    item_rmse.append(r.rmse)
                    item_mae.append(r.mae)
    
    # Create RMSE line chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, item_rmse, marker="o", label="ItemCF", linewidth=2, markersize=8, color="blue")
    ax.plot(k_values, user_rmse, marker="o", label="UserCF", linewidth=2, markersize=8, color="orange")
    ax.set_title("RMSE vs k (cosine)", fontsize=14, fontweight="bold")
    ax.set_xlabel("k neighbors", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)
    ax.set_xticks(k_values)
    plt.tight_layout()
    output_path = RESULTS_DIR / "task2_neighbors_rmse.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Task2] Saved RMSE line chart to {output_path}")
    
    # Create MAE line chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, item_mae, marker="o", label="ItemCF", linewidth=2, markersize=8, color="blue")
    ax.plot(k_values, user_mae, marker="o", label="UserCF", linewidth=2, markersize=8, color="orange")
    ax.set_title("MAE vs k", fontsize=14, fontweight="bold")
    ax.set_xlabel("k neighbors", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)
    ax.set_xticks(k_values)
    plt.tight_layout()
    output_path = RESULTS_DIR / "task2_neighbors_mae.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Task2] Saved MAE line chart to {output_path}")

