"""F5 proxy — cross-example latent-KV similarity analysis.

Since the live F5 swap experiment is blocked on GPU, this proxy uses the
existing final-layer latent-KV dumps to characterize how much latent KV
varies across examples. Specifically:

  - Compute pairwise cosine similarity of flattened latent-KV vectors
    across the first 200 GSM8k examples.
  - Median + percentiles of pairwise similarity.
  - Variance-explained by top-k principal components of the stack.

If pairwise cosine similarity is near 1.0 for the majority of pairs AND
variance is concentrated in very few PCs, the latent KV is near-identical
across examples (supports the inert-template hypothesis). If similarity is
low and variance is spread, latent KV is per-example.
"""

import json
from pathlib import Path

import numpy as np

DUMP_DIR = Path(
    "/Users/jrauvola/Desktop/Latent_Reasoning_Project/research_findings/kv_pca/"
    "qwen3_4b_codi_bf16_kv_latent_detach_last_2/gsm8k/numlatent_8"
)
OUT_JSON = Path(
    "/Users/jrauvola/Desktop/Latent_Reasoning_Project/research_findings/inert_latent_F5_proxy.json"
)


def main() -> None:
    files = sorted(DUMP_DIR.glob("kv_example_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    assert files, f"no KV dumps at {DUMP_DIR}"
    print(f"loading {len(files)} dumps from {DUMP_DIR}")

    stacks = []
    for f in files:
        arr = np.load(f)  # expected shape [num_latent, 2, heads, head_dim] (final-layer, single batch slice)
        stacks.append(arr.reshape(-1))  # flatten each example
    X = np.stack(stacks, axis=0)  # [N, D]
    print(f"stacked shape {X.shape} dtype={X.dtype}")

    # Normalize and compute pairwise cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-12)
    sims = Xn @ Xn.T  # [N, N]
    triu = np.triu_indices(sims.shape[0], k=1)
    pair_sims = sims[triu]
    print(f"pairwise cosine similarity: median={np.median(pair_sims):.4f} "
          f"p5={np.percentile(pair_sims, 5):.4f} p95={np.percentile(pair_sims, 95):.4f}")

    # PCA on centered X
    Xc = X - X.mean(axis=0, keepdims=True)
    # Use SVD for PCA
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Variance explained
    var = S**2
    var_ratio = var / var.sum()
    cum_ratio = np.cumsum(var_ratio)
    # Number of PCs to reach 80% variance
    k80 = int(np.searchsorted(cum_ratio, 0.80)) + 1
    k95 = int(np.searchsorted(cum_ratio, 0.95)) + 1

    top10 = var_ratio[:10].tolist()
    print(f"top-1 PC variance: {var_ratio[0]:.4f}; top-3: {var_ratio[:3].sum():.4f}; top-10: {var_ratio[:10].sum():.4f}")
    print(f"PCs to reach 80%/95% variance: {k80}/{k95}")

    # Also: distance to centroid (how tight is the cluster?)
    centroid = X.mean(axis=0, keepdims=True)
    diffs = X - centroid
    dist = np.linalg.norm(diffs, axis=1)
    mean_norm = np.linalg.norm(X, axis=1).mean()
    print(f"mean |x|={mean_norm:.4f}  mean |x - centroid|={dist.mean():.4f}  "
          f"ratio={dist.mean() / max(mean_norm, 1e-9):.4f}")

    out = {
        "n_examples": int(X.shape[0]),
        "flattened_dim": int(X.shape[1]),
        "pair_cosine_median": float(np.median(pair_sims)),
        "pair_cosine_p5": float(np.percentile(pair_sims, 5)),
        "pair_cosine_p50": float(np.percentile(pair_sims, 50)),
        "pair_cosine_p95": float(np.percentile(pair_sims, 95)),
        "pc_var_ratio_top10": top10,
        "pc_count_80pct_var": int(k80),
        "pc_count_95pct_var": int(k95),
        "mean_norm": float(mean_norm),
        "mean_distance_to_centroid": float(dist.mean()),
        "distance_to_centroid_over_norm": float(dist.mean() / max(mean_norm, 1e-9)),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
