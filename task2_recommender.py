#!/usr/bin/env python3
"""Recommender system experiments: PMF, UserCF, ItemCF evaluation with 5-fold CV."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from surprise import Dataset, KNNBasic, Reader, SVD, accuracy
from surprise.model_selection import KFold

from task2_plots import plot_neighbor_sweep, plot_similarity_bar_charts

DATA_FILE = Path("archive") / "ratings_small.csv"
RESULTS_DIR = Path("results")


@dataclass
class CVResult:
    """Cross-validation result container."""
    model: str
    detail: str
    mae: float
    rmse: float


def ensure_results_dir() -> None:
    """Create results directory."""
    RESULTS_DIR.mkdir(exist_ok=True)


def load_surprise_dataset(ratings_file: Path) -> Dataset:
    """Load ratings dataset."""
    reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)
    return Dataset.load_from_file(str(ratings_file), reader=reader)


def evaluate_algorithm(algo, data: Dataset, n_splits: int = 5, seed: int = 7) -> Tuple[float, float]:
    """Evaluate algorithm with k-fold CV."""
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    mae_scores, rmse_scores = [], []
    for fold_idx, (trainset, testset) in enumerate(kf.split(data), 1):
        algo.fit(trainset)
        predictions = algo.test(testset)
        mae_scores.append(accuracy.mae(predictions, verbose=False))
        rmse_scores.append(accuracy.rmse(predictions, verbose=False))
        if fold_idx < n_splits:
            print(f"  Completed fold {fold_idx}/{n_splits}...", end="\r")
    print(f"  Completed all {n_splits} folds.      ")
    return float(np.mean(mae_scores)), float(np.mean(rmse_scores))


def baseline_models(data: Dataset, seed: int) -> List[CVResult]:
    """Evaluate baseline models: PMF, UserCF, ItemCF."""
    algos = {
        "PMF": SVD(n_factors=50, biased=False, random_state=seed),
        "UserCF": KNNBasic(k=40, min_k=1, sim_options={"name": "cosine", "user_based": True}),
        "ItemCF": KNNBasic(k=40, min_k=1, sim_options={"name": "cosine", "user_based": False}),
    }
    results = []
    for name, algo in algos.items():
        print(f"[Task2] Evaluating {name} with 5-fold CV...")
        mae, rmse = evaluate_algorithm(algo, data, seed=seed)
        results.append(CVResult(model=name, detail="baseline", mae=mae, rmse=rmse))
    return results


def similarity_sweep(data: Dataset, seed: int) -> List[CVResult]:
    """Test different similarity metrics."""
    sims = ["cosine", "msd", "pearson"]
    results = []
    for sim_name in sims:
        user_algo = KNNBasic(k=40, min_k=1, sim_options={"name": sim_name, "user_based": True})
        item_algo = KNNBasic(k=40, min_k=1, sim_options={"name": sim_name, "user_based": False})
        print(f"[Task2] Similarity sweep: user-based {sim_name}")
        u_mae, u_rmse = evaluate_algorithm(user_algo, data, seed=seed)
        print(f"[Task2] Similarity sweep: item-based {sim_name}")
        i_mae, i_rmse = evaluate_algorithm(item_algo, data, seed=seed)
        results.append(CVResult(model="UserCF", detail=f"sim={sim_name}", mae=u_mae, rmse=u_rmse))
        results.append(CVResult(model="ItemCF", detail=f"sim={sim_name}", mae=i_mae, rmse=i_rmse))
    return results


def neighbor_sweep(data: Dataset, seed: int, neighbors: Iterable[int]) -> List[CVResult]:
    """Test different neighbor counts."""
    results = []
    k_values = list(neighbors)
    total = len(k_values) * 2
    current = 0
    
    for k in k_values:
        user_algo = KNNBasic(k=k, min_k=1, sim_options={"name": "cosine", "user_based": True})
        item_algo = KNNBasic(k=k, min_k=1, sim_options={"name": "cosine", "user_based": False})
        current += 1
        print(f"[Task2] Neighbor sweep ({current}/{total}): user-based k={k} (this may take a while...)")
        u_mae, u_rmse = evaluate_algorithm(user_algo, data, seed=seed)
        current += 1
        print(f"[Task2] Neighbor sweep ({current}/{total}): item-based k={k} (this may take a while...)")
        i_mae, i_rmse = evaluate_algorithm(item_algo, data, seed=seed)
        results.append(CVResult(model="UserCF", detail=f"k={k}", mae=u_mae, rmse=u_rmse))
        results.append(CVResult(model="ItemCF", detail=f"k={k}", mae=i_mae, rmse=i_rmse))
    return results


def save_results(tag: str, results: List[CVResult]) -> None:
    """Save results to CSV and JSON."""
    ensure_results_dir()
    csv_path = RESULTS_DIR / f"task2_{tag}.csv"
    json_path = RESULTS_DIR / f"task2_{tag}.json"
    if not results:
        return
    header = list(asdict(results[0]).keys())
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in results:
            fh.write(",".join(str(asdict(row)[col]) for col in header) + "\n")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    print(f"[Task2] Saved {tag} results to {csv_path}")


def save_all_results_csv(
    baseline: List[CVResult],
    similarity: List[CVResult],
    neighbors: List[CVResult],
) -> None:
    """Save combined results."""
    ensure_results_dir()
    all_results = baseline + similarity + neighbors
    if not all_results:
        return
    csv_path = RESULTS_DIR / "task2_results.csv"
    header = list(asdict(all_results[0]).keys())
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in all_results:
            fh.write(",".join(str(asdict(row)[col]) for col in header) + "\n")
    print(f"[Task2] Saved all results to {csv_path}")


def compare_models(baseline: List[CVResult]) -> None:
    """Compare models and find best."""
    print("\n" + "="*70)
    print("[Task2] Model Comparison (Requirement 3.d)")
    print("="*70)
    
    pmf_result = next((r for r in baseline if r.model == "PMF"), None)
    user_result = next((r for r in baseline if r.model == "UserCF"), None)
    item_result = next((r for r in baseline if r.model == "ItemCF"), None)
    
    if not all([pmf_result, user_result, item_result]):
        print("[Task2] Warning: Missing model results for comparison")
        return
    
    print("\nAverage Performance (5-fold CV):")
    print("-" * 70)
    print(f"{'Model':<15} {'MAE':<15} {'RMSE':<15}")
    print("-" * 70)
    print(f"{'PMF':<15} {pmf_result.mae:<15.6f} {pmf_result.rmse:<15.6f}")
    print(f"{'UserCF':<15} {user_result.mae:<15.6f} {user_result.rmse:<15.6f}")
    print(f"{'ItemCF':<15} {item_result.mae:<15.6f} {item_result.rmse:<15.6f}")
    print("-" * 70)
    
    models_rmse = [("PMF", pmf_result.rmse), ("UserCF", user_result.rmse), ("ItemCF", item_result.rmse)]
    models_mae = [("PMF", pmf_result.mae), ("UserCF", user_result.mae), ("ItemCF", item_result.mae)]
    best_rmse = min(models_rmse, key=lambda x: x[1])
    best_mae = min(models_mae, key=lambda x: x[1])
    
    print(f"\nBest Model by RMSE: {best_rmse[0]} (RMSE = {best_rmse[1]:.6f})")
    print(f"Best Model by MAE:   {best_mae[0]} (MAE = {best_mae[1]:.6f})")
    
    if best_rmse[0] == best_mae[0]:
        print(f"\n>>> Overall Best Model: {best_rmse[0]} <<<")
        print(f"   (Best in both RMSE and MAE)")
    else:
        print(f"\n>>> Best Model: {best_rmse[0]} (by RMSE) or {best_mae[0]} (by MAE) <<<")
        print(f"   (Different models excel in different metrics)")
    
    print("="*70 + "\n")


def find_best_k(results: List[CVResult], neighbors: List[int]) -> None:
    """Find best K for UserCF and ItemCF."""
    print("\n" + "="*70)
    print("[Task2] Best Number of Neighbors Analysis (Requirement 3.g)")
    print("="*70)
    
    k_values = sorted(neighbors)
    
    usercf_results = [(k, r.rmse) for k in k_values for r in results 
                      if r.detail == f"k={k}" and r.model == "UserCF"]
    if usercf_results:
        best_usercf = min(usercf_results, key=lambda x: x[1])
        print(f"\nUser-based Collaborative Filtering:")
        print(f"  Best K: {best_usercf[0]} (RMSE = {best_usercf[1]:.6f})")
        print(f"  All K values and RMSE:")
        for k, rmse in sorted(usercf_results):
            marker = " <-- BEST" if k == best_usercf[0] else ""
            print(f"    K={k:3d}: RMSE = {rmse:.6f}{marker}")
    else:
        print("\nUser-based Collaborative Filtering: No results found")
        best_usercf = None
    
    itemcf_results = [(k, r.rmse) for k in k_values for r in results 
                     if r.detail == f"k={k}" and r.model == "ItemCF"]
    if itemcf_results:
        best_itemcf = min(itemcf_results, key=lambda x: x[1])
        print(f"\nItem-based Collaborative Filtering:")
        print(f"  Best K: {best_itemcf[0]} (RMSE = {best_itemcf[1]:.6f})")
        print(f"  All K values and RMSE:")
        for k, rmse in sorted(itemcf_results):
            marker = " <-- BEST" if k == best_itemcf[0] else ""
            print(f"    K={k:3d}: RMSE = {rmse:.6f}{marker}")
    else:
        print("\nItem-based Collaborative Filtering: No results found")
        best_itemcf = None
    
    if best_usercf and best_itemcf:
        print(f"\n{'='*70}")
        if best_usercf[0] == best_itemcf[0]:
            print(f">>> Best K is the SAME for both: K = {best_usercf[0]} <<<")
        else:
            print(f">>> Best K is DIFFERENT:")
            print(f"   UserCF: K = {best_usercf[0]} (RMSE = {best_usercf[1]:.6f})")
            print(f"   ItemCF: K = {best_itemcf[0]} (RMSE = {best_itemcf[1]:.6f}) <<<")
    
    print("="*70 + "\n")


def main() -> None:
    """Main execution flow."""
    parser = argparse.ArgumentParser(description="Run Task 2 recommender experiments.")
    parser.add_argument("--ratings-path", type=Path, default=DATA_FILE,
                       help="Path to ratings_small.csv (default: archive/ratings_small.csv)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--neighbors", type=int, nargs="+", default=[10, 20, 40, 60],
                       help="Neighbor counts for the sweep (default: 10 20 40 60)")
    args = parser.parse_args()

    data = load_surprise_dataset(args.ratings_path)

    baseline = baseline_models(data, args.seed)
    save_results("baseline", baseline)
    compare_models(baseline)

    sim_results = similarity_sweep(data, args.seed)
    save_results("similarity", sim_results)
    plot_similarity_bar_charts(sim_results)

    neighbor_results = neighbor_sweep(data, args.seed, args.neighbors)
    save_results("neighbors", neighbor_results)
    plot_neighbor_sweep(neighbor_results, args.neighbors)
    find_best_k(neighbor_results, args.neighbors)

    save_all_results_csv(baseline, sim_results, neighbor_results)
    print("[Task2] All experiments complete.")


if __name__ == "__main__":
    main()
