
# Task1
import numpy as np
import pandas as pd
import time
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# norm calc
def _row_norms(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.sqrt((X * X).sum(axis=1) + eps)
# l2 grid
def euclidean_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # dist form
    x2 = (X * X).sum(axis=1, keepdims=True)         
    c2 = (C * C).sum(axis=1, keepdims=True).T       
    d2 = x2 - 2 * X @ C.T + c2                      
    np.maximum(d2, 0.0, out=d2)
    return d2

# cosine grid
def cosine_dists(X: np.ndarray, C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Xn = X / (_row_norms(X, eps=eps)[:, None])
    Cn = C / (_row_norms(C, eps=eps)[:, None])
    sim = Xn @ Cn.T
    return 1.0 - sim

# jaccard grid
def generalized_jaccard_dists(X: np.ndarray, C: np.ndarray, chunk_size: int = 2000) -> np.ndarray:
    N, D = X.shape
    K = C.shape[0]
    out = np.empty((N, K), dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Xchunk = X[start:end]

        for k in range(K):
            ck = C[k] 
            mins = np.minimum(Xchunk, ck)  
            maxs = np.maximum(Xchunk, ck)
            num = mins.sum(axis=1)   
            den = maxs.sum(axis=1) + 1e-12
            sim = num / den
            out[start:end, k] = 1.0 - sim
    return out

#seed util
#smart seeds
def kmeans_plus_plus_init(X: np.ndarray, K: int, dist_fn) -> np.ndarray:
    N = X.shape[0]
    rng = np.random.default_rng(42)
    first = rng.integers(0, N)
    centroids = [X[first]]
    for _ in range(1, K):
        C = np.stack(centroids, axis=0)
        d = dist_fn(X, C)
        dd = d if dist_fn is euclidean_sq_dists else (d * d)
        mind = dd.min(axis=1)
        probs = mind / (mind.sum() + 1e-12)
        idx = rng.choice(N, p=probs)
        centroids.append(X[idx])
    return np.stack(centroids, axis=0)

# core bits
class KMeansVariant:
    def __init__(
        self,
        K: int,
        metric: str = "euclidean",
        max_iter: int = 500,
        stop_mode: str = "combo", 
        random_state: int = 42,
    ):
        assert metric in ("euclidean", "cosine", "jaccard")
        assert stop_mode in ("combo", "steady", "worse", "cap")
        self.K = K
        self.metric = metric
        self.max_iter = max_iter
        self.stop_mode = stop_mode
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None  # SSE val
        self.n_iter_: int = 0
        self.history_: List[float] = []
        self.fit_time_: float = 0.0

    #metric ops
    def _dist_fn(self):
        if self.metric == "euclidean":
            return euclidean_sq_dists
        elif self.metric == "cosine":
            return cosine_dists
        else:
            return generalized_jaccard_dists

    def _postprocess_centroids(self, C: np.ndarray) -> np.ndarray:
        """For cosine (spherical k-means), normalize centroids to unit length. Clip nonnegativity for jaccard."""
        if self.metric == "cosine":
            norms = np.sqrt((C * C).sum(axis=1, keepdims=True)) + 1e-12
            C = C / norms
        elif self.metric == "jaccard":
            C = np.clip(C, 0.0, None)  # keep >=0
        return C

    def _compute_sse(self, X: np.ndarray, C: np.ndarray, assign: np.ndarray) -> float:
        """Sum of squared distances under the chosen metric."""
        # per dist
        dists = np.empty(X.shape[0], dtype=np.float64)
        if self.metric in ("euclidean", "cosine"):
            D = self._dist_fn()(X, C)  # (N, K)
            d = D[np.arange(X.shape[0]), assign]
            dists[:] = d
        else:
            for k in range(self.K):
                idx = np.where(assign == k)[0]
                if idx.size == 0:
                    continue
                Dk = generalized_jaccard_dists(X[idx], C[k:k+1])[:, 0]
                dists[idx] = Dk
        #SSE def
        return float(np.sum(dists ** 2))

    # fit loop
    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        dist_fn = self._dist_fn()
        # kick seeds
        C = kmeans_plus_plus_init(X, self.K, dist_fn)

        start_time = time.time()
        last_sse = np.inf
        prev_assign = None
        for it in range(1, self.max_iter + 1):
            # assign step
            D = dist_fn(X, C)  
            assign = D.argmin(axis=1)

            # mean step
            C_new = np.zeros_like(C)
            for k in range(self.K):
                mask = (assign == k)
                if not np.any(mask):
                    C_new[k] = X[rng.integers(0, X.shape[0])]
                else:
                    C_new[k] = X[mask].mean(axis=0)

            C_new = self._postprocess_centroids(C_new)

            #SSE check
            sse = self._compute_sse(X, C_new, assign)
            self.history_.append(sse)

            #stop logic
            no_change = np.allclose(C_new, C, atol=1e-8)
            sse_increase = sse > last_sse - 1e-12

            stop_any = (no_change or sse_increase or (it >= self.max_iter))

            if self.stop_mode == "combo":
                if stop_any:
                    if sse_increase:
                        last_sse = last_sse
                    else:
                        #Use new centroids
                        C = C_new
                        last_sse = sse
                    self.n_iter_ = it
                    break
            elif self.stop_mode == "steady":
                if no_change or (it >= self.max_iter):
                    C = C_new
                    last_sse = sse
                    self.n_iter_ = it
                    break
            elif self.stop_mode == "worse":
                if sse_increase or (it >= self.max_iter):
                    if sse_increase:
                        last_sse = last_sse
                    else:
                        #new centroids
                        C = C_new
                        last_sse = sse
                    self.n_iter_ = it
                    break
            elif self.stop_mode == "cap":
                if it == self.max_iter:
                    C = C_new
                    last_sse = sse
                    self.n_iter_ = it
                    break

            #update centroids
            if not sse_increase:
                C = C_new
                last_sse = sse
                prev_assign = assign.copy()
            else:
                # SSE increased
                prev_assign = prev_assign if prev_assign is not None else assign.copy()

        end_time = time.time()

        #final pass
        D = dist_fn(X, C)
        assign = D.argmin(axis=1)

        self.centroids_ = C
        self.labels_ = assign
        self.inertia_ = self._compute_sse(X, C, assign)
        self.fit_time_ = end_time - start_time
        return self

# vote score
def majority_vote_accuracy(pred_clusters: np.ndarray, true_labels: np.ndarray, K: int) -> float:
    label_map: Dict[int, int] = {}
    total_correct = 0
    for k in range(K):
        idx = np.where(pred_clusters == k)[0]
        if idx.size == 0:
            continue
        votes = Counter(true_labels[idx])
        top_label, cnt = votes.most_common(1)[0]
        label_map[k] = top_label
        total_correct += cnt
    return total_correct / true_labels.shape[0]

def run_variant(X: np.ndarray, y: np.ndarray, metric: str, K: int, stop_mode: str, max_iter: int = 500):
    km = KMeansVariant(K=K, metric=metric, max_iter=max_iter, stop_mode=stop_mode, random_state=42).fit(X)
    acc = majority_vote_accuracy(km.labels_, y, K)
    return {
        "metric": metric,
        "stop_mode": stop_mode,
        "SSE": km.inertia_,
        "accuracy": acc,
        "n_iter": km.n_iter_,
        "fit_time_sec": km.fit_time_,
    }

#main
def main():
    # load data sets
    base_path = Path(__file__).parent
    data_path = base_path / "kmeans_data" / "data.csv"
    label_path = base_path / "kmeans_data" / "label.csv"
    X = pd.read_csv(data_path, header=None).values.astype(np.float64)
    y = pd.read_csv(label_path, header=None).values.squeeze()
    if y.ndim > 1:
        y = y[:, 0]

    # scale data
    X = X - X.min()
    x_max = X.max()
    if x_max > 0:
        X = X / x_max

    # set k
    K = int(len(np.unique(y)))

    results: List[Dict] = []

    # Q1 Compare SSEs 
    q1_results = []
    for metric in ("euclidean", "cosine", "jaccard"):
        res = run_variant(X, y, metric=metric, K=K, stop_mode="combo", max_iter=500)
        q1_results.append({**res, "question": "Q1/Q3 (combined stop)"})
        results.append(q1_results[-1])
        print(f"[Q1/Q3] {metric:9s} | SSE={res['SSE']:.4f} | acc={res['accuracy']:.4f} | "
              f"iters={res['n_iter']} | time={res['fit_time_sec']:.2f}s")

    # Q2 accuracy comparison
    for res in q1_results:
        results.append({**res, "question": "Q2 (accuracy w/ majority vote)"})
        print(f"[Q2]     {res['metric']:9s} | acc={res['accuracy']:.4f} | SSE={res['SSE']:.4f} | "
              f"iters={res['n_iter']} | time={res['fit_time_sec']:.2f}s")

    # Q3 note
    # show rank
    subset_q3 = [r for r in results if r["question"] == "Q1/Q3 (combined stop)"]
    q3_sorted_iters = sorted(subset_q3, key=lambda d: d["n_iter"], reverse=True)
    q3_sorted_time = sorted(subset_q3, key=lambda d: d["fit_time_sec"], reverse=True)
    print("\n[Q3] Iterations (desc):", " > ".join([f"{r['metric']}({r['n_iter']})" for r in q3_sorted_iters]))
    print("[Q3] Time (desc):      ", " > ".join([f'{r["metric"]}({r["fit_time_sec"]:.2f}s)' for r in q3_sorted_time]))

    # Q4 loop
    for stop_mode in ("steady", "worse", "cap"):
        for metric in ("euclidean", "cosine", "jaccard"):
            res = run_variant(X, y, metric=metric, K=K, stop_mode=stop_mode,
                              max_iter=100 if stop_mode == "cap" else 500)
            results.append({**res, "question": f"Q4 ({stop_mode})"})
            print(f"[Q4:{stop_mode:18s}] {metric:9s} | SSE={res['SSE']:.4f} | acc={res['accuracy']:.4f} | "
                  f"iters={res['n_iter']} | time={res['fit_time_sec']:.2f}s")

    # save csv
    df = pd.DataFrame(results)
    output_dir = base_path / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "task1_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved summary to {output_file}")

if __name__ == "__main__":
    main()
