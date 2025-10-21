"""Fractal analysis utilities for social network coverage (box-counting, per-seed fits, aggregation).

A fractal is a pattern that displays self-similarity across scales. In network analysis,
a fractal (or scaling) exponent D can be estimated by measuring how the number of nodes
covered within radius r, N(r), grows with r. The usual empirical relation is:

    N(r) ~ r^D   <=>   log N(r) ~ D * log r

This module provides helpers to:
- run per-seed BFS/coverage up to a maximum radius,
- perform log-log fits per seed and on aggregated curves (median/mean),
- compute a robust median-of-seeds estimator excluding saturated radii and poor fits,
- generate diagnostic plots.

The functions use a pragmatic approach intended for exploratory analysis; see the
documentation comments in each function for usage details and returned keys.
"""

import random
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

def fit_fractal_dimension(box_sizes: List[int], box_counts: List[int], min_points: int = 3) -> Tuple[float, float]:
    """
    Estimate fractal dimension D and coefficient of determination R^2 from box-counting data.

    Performs a linear regression on (log(box_size), log(box_count)). Uses numpy.polyfit
    fallback when scipy is not available.

    Args:
        box_sizes (List[int]): increasing list of integer radii (r values).
        box_counts (List[int]): corresponding cumulative counts N(r) for each radius.
        min_points (int): minimum number of valid (positive) points required to fit.

    Returns:
        Tuple[float, float]:
            (D, r2) where D is the slope of the log-log fit and r2 is the coefficient
            of determination (in log space). Returns (nan, nan) if insufficient data.

    Example:
        >>> D, r2 = fit_fractal_dimension([1,2,3], [10,40,100])
    """

    # filter positive finite points (exclude zeros and negatives because log undefined)
    xs = []
    ys = []
    for r, c in zip(box_sizes, box_counts):
        # r: radius value; c: covered nodes at that radius
        if c is None:
            continue
        try:
            cval = float(c)
        except Exception:
            continue
        if cval > 0 and r > 0:
            xs.append(r)
            ys.append(cval)

    # need at least min_points to fit
    if len(xs) < min_points:
        return float('nan'), float('nan')

    # perform linear fit on natural logarithms
    import numpy as _np
    lx = _np.log(_np.array(xs, dtype=float))   # log(r)
    ly = _np.log(_np.array(ys, dtype=float))   # log(N)

    # numpy.polyfit performs least squares fit on (lx, ly); coeffs[0] is slope (D), coeffs[1] intercept (log C)
    coeffs = _np.polyfit(lx, ly, 1)
    slope = float(coeffs[0])      # estimated D
    intercept = float(coeffs[1])  # estimated log C

    # predicted values in log-space and R^2 calculation
    pred = slope * lx + intercept
    ss_res = float(_np.sum((ly - pred) ** 2))           # residual sum of squares (log space)
    ss_tot = float(_np.sum((ly - ly.mean()) ** 2))     # total sum of squares (log space)

    # handle degenerate case: zero variance in ly -> non-informative fit
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return slope, r2

def calculate_fractal_dimension(
    driver,
    max_box_size: int = 5,
    sample_size: int = 0,
    graph_name: str = "userGraph",
    max_start_nodes_for_full: int = 500,
    generate_plots: bool = False,
    plot_dir: str = "artifacts/fractal_plots",
    sample_seeds_for_plot: int = 6,
    agg_method: str = "median",  # "median" or "mean"
) -> Dict[str, Any]:
    """
    Compute fractal (box-counting) dimension for a Neo4j graph projection.

    Two modes:
    - Sample mode: when sample_size > 0 compute per-sample coverage (e.g. per tweet author).
    - Full mode: sample users (or use all if small) and compute per-seed BFS coverage up to max_box_size.

    The function aggregates per-seed coverages (median or mean) and fits an aggregated log-log
    curve. It also computes per-seed fits and returns summary statistics.

    Args:
        driver:
            Neo4j driver object with .session() context manager.
        max_box_size (int):
            Maximum radius (box size) to query with BFS.
        sample_size (int):
            If >0 run sample mode over tweets; otherwise run full network mode.
        graph_name (str):
            Name of the GDS graph projection used by gds.bfs.stream.
        max_start_nodes_for_full (int):
            Maximum number of start nodes to sample when analyzing the full network.
        generate_plots (bool):
            If True generate diagnostic plots (requires matplotlib).
        plot_dir (str):
            Destination directory for plots.
        sample_seeds_for_plot (int):
            Number of per-seed curves to sample when plotting per-seed coverage.
        agg_method (str):
            'median' or 'mean' aggregation across seeds for the aggregated curve.

    Returns:
        Dict[str, Any]:
            Result dictionary containing keys described in the module docstring,
            including aggregated D/r2, per-seed Ds, per-seed r2s, and optional 'median_filtered'.

    Example:
        >>> res = calculate_fractal_dimension(driver, max_box_size=6, generate_plots=True, agg_method='median')
    """

    # Relationship projection definition 
    rel_proj = "{ RETWEETED_FROM: { type: 'RETWEETED_FROM', orientation: 'UNDIRECTED' } }"

    # Ensure GDS graph projection exists
    with driver.session() as session:
        exists_row = session.run(f"CALL gds.graph.exists('{graph_name}') YIELD exists RETURN exists").single()
        exists = bool(exists_row["exists"]) if exists_row and "exists" in exists_row else False
        if not exists:
            print(f"Creating GDS graph projection '{graph_name}' ...")
            session.run(f"CALL gds.graph.project('{graph_name}', 'User', {rel_proj})")

    # --- SAMPLE MODE: per tweet (seed = author) ---
    if sample_size and sample_size > 0:
        with driver.session() as session:
            tweet_rows = session.run("MATCH (t:Tweet) WHERE t.text IS NOT NULL RETURN t.tweet_id AS tweet_id")
            tweet_ids = [r["tweet_id"] for r in tweet_rows]

        if not tweet_ids:
            print("No tweets with text found.")
            return {}

        sampled = random.sample(tweet_ids, min(sample_size, len(tweet_ids)))
        results: Dict[str, Any] = {}

        with driver.session() as session:
            for tid in sampled:
                # recupera id interno dell'autore: usa relazione User -[:CREATES]-> Tweet
                rec = session.run(
                    "MATCH (u:User)-[:CREATES]->(t:Tweet {tweet_id: $tid}) RETURN id(u) AS nid",
                    tid=tid,
                ).single()
                if not rec or rec.get("nid") is None:
                    results[tid] = {"D": float("nan"), "r2": float("nan"), "box_sizes": [], "box_counts": []}
                    continue
                start_nid = rec["nid"]

                box_sizes: List[int] = []
                box_counts: List[int] = []

                # per ogni raggio chiamare gds.bfs.stream con sourceNode singolo
                for box in range(1, max_box_size + 1):
                    bfs_query = f"""
                    CALL gds.bfs.stream(
                      '{graph_name}',
                      {{ sourceNode: $source_nid, maxDepth: $maxDepth }}
                    )
                    YIELD sourceNode, nodeIds
                    UNWIND nodeIds AS nodeId
                    RETURN count(DISTINCT nodeId) AS covered
                    """
                    row = session.run(bfs_query, source_nid=start_nid, maxDepth=box).single()
                    covered = int(row["covered"]) if row and row.get("covered") is not None else 0
                    if covered == 0:
                        break
                    box_sizes.append(box)
                    box_counts.append(covered)

                D, r2 = fit_fractal_dimension(box_sizes, box_counts)
                results[tid] = {"D": D, "r2": r2, "box_sizes": box_sizes, "box_counts": box_counts}
                if np.isfinite(D):
                    print(f"Tweet {tid}: D={D:.4f} (R^2={r2:.3f}), points={len(box_sizes)}")
                else:
                    print(f"Tweet {tid}: insufficient points for fit (points={len(box_sizes)})")

        return results

    # --- FULL NETWORK MODE ---
    else:
        # --- FULL NETWORK MODE : per-seed BFS + aggregation ---
        with driver.session() as session:
            starts_rows = session.run("MATCH (u:User) RETURN id(u) AS nid")
            starts = [r["nid"] for r in starts_rows]

        if not starts:
            print("No users found in DB.")
            return {"D": float("nan"), "r2": float("nan"), "box_sizes": [], "box_counts": []}

        # Sample start nodes if too many
        if len(starts) > max_start_nodes_for_full:
            print(f"Found {len(starts)} users — sampling {max_start_nodes_for_full} start nodes for efficiency.")
            starts = random.sample(starts, max_start_nodes_for_full)

        # per-seed counts: list of lists; each inner list = counts per box for one seed
        per_seed_counts: List[List[int]] = []

        with driver.session() as session:
            for start_nid in starts:
                seed_counts: List[int] = []
                for box in range(1, max_box_size + 1):
                    bfs_query = f"""
                    CALL gds.bfs.stream(
                      '{graph_name}',
                      {{ sourceNode: $source_nid, maxDepth: $maxDepth }}
                    )
                    YIELD nodeIds
                    UNWIND nodeIds AS nodeId
                    RETURN count(DISTINCT nodeId) AS covered
                    """
                    row = session.run(bfs_query, source_nid=start_nid, maxDepth=box).single()
                    covered = int(row["covered"]) if row and row.get("covered") is not None else 0
                    if covered == 0:
                        break
                    seed_counts.append(covered)
                if seed_counts:
                    per_seed_counts.append(seed_counts)

        # Align lengths (some seeds may have fewer points); build valid box_sizes
        if not per_seed_counts:
            print("No per-seed coverage found.")
            return {"D": float("nan"), "r2": float("nan"), "box_sizes": [], "box_counts": []}

        # compute aggregated counts per radius: use mean and median (ignore seeds without value for that box)
        import statistics
        max_len = max(len(s) for s in per_seed_counts)
        agg_mean: List[float] = []
        agg_median: List[float] = []
        box_sizes: List[int] = []
        for i in range(max_len):
            vals = [s[i] for s in per_seed_counts if len(s) > i]
            if not vals:
                break
            agg_mean.append(float(sum(vals)) / len(vals))
            agg_median.append(float(statistics.median(vals)))
            box_sizes.append(i + 1)

        # fit fractal on aggregated curve (choose median or mean by agg_method)
        if agg_method.lower() == "mean":
            box_counts_for_fit = [int(x) for x in agg_mean]
        else:
            box_counts_for_fit = [int(x) for x in agg_median]

        D, r2 = fit_fractal_dimension(box_sizes, box_counts_for_fit)

        # compute per-seed D distribution (use fit on each seed with >=3 points)
        per_seed_Ds: List[float] = []
        per_seed_r2s: List[float] = []
        for s_counts in per_seed_counts:
            d, r = fit_fractal_dimension(list(range(1, len(s_counts)+1)), s_counts)
            per_seed_Ds.append(d)
            per_seed_r2s.append(r)

        # report
        print(f"Aggregated box_sizes: {box_sizes}")
        print(f"Aggregated (median) box_counts: {box_counts_for_fit}")
        if np.isfinite(D):
            print(f"Network (aggregated median): D={D:.4f} (R^2={r2:.3f}), points={len(box_sizes)}")
        else:
            print(f"Network: insufficient points for fit (points={len(box_sizes)})")

        # summary stats per-seed (filtra seed con r2 > threshold)
        good_threshold = 0.8
        good_Ds = [
            d
            for d, r in zip(per_seed_Ds, per_seed_r2s)
            if np.isfinite(d) and np.isfinite(r) and (r > good_threshold)
        ]
        median_D = float(statistics.median(good_Ds)) if good_Ds else float("nan")
        mean_D = float(sum(good_Ds) / len(good_Ds)) if good_Ds else float("nan")
        per_seed_good_count = len(good_Ds)

        result = {
            "D": D,
            "r2": r2,
            "box_sizes": box_sizes,
            "box_counts_median": box_counts_for_fit if agg_method == "median" else box_counts_for_fit,
            "box_counts_mean": agg_mean,
            "per_seed_counts": per_seed_counts,
            "per_seed_Ds": per_seed_Ds,
            "per_seed_r2s": per_seed_r2s,
            "per_seed_good_count": per_seed_good_count,
            "per_seed_count": len(per_seed_counts),
            "per_seed_D_median": median_D,
            "per_seed_D_mean": mean_D,
        }

    # --- median-filtered estimation (exclude saturation + r2 filter) ---
    def median_filtered_D(
        res: Dict[str, Any],
        total_nodes: Optional[int] = None,
        sat_thresh: float = 0.90,
        r2_thresh: float = 0.8,
        rmin: int = 2,
        min_points: int = 3,
        n_boot: int = 1000,
    ) -> Dict[str, Any]:
        """
        Robust estimation of median per-seed fractal dimension excluding saturated radii.

        This helper selects radii that are not saturated (aggregated median coverage
        below sat_thresh * total_nodes) and r >= rmin, then:
        - fits log(N) ~ D * log(r) per seed using only those radii where the seed
          has data,
        - keeps seeds with R^2 >= r2_thresh and at least `min_points`,
        - returns the median of retained per-seed D values and a bootstrap 95% CI.

        Args:
            res (Dict[str, Any]):
                Result dictionary produced by calculate_fractal_dimension. Must contain
                'per_seed_counts' (list of lists) and 'box_counts_median' / 'box_sizes'.
            total_nodes (Optional[int]):
                Optional total number of nodes used to define saturation. If None the
                function will estimate total_nodes from per-seed maxima or aggregated median.
            sat_thresh (float):
                Fraction of total_nodes above which a radius is considered saturated (e.g. 0.9).
            r2_thresh (float):
                Minimum R^2 for a per-seed fit to be considered reliable.
            rmin (int):
                Minimum radius to include (avoids r=1 noise).
            min_points (int):
                Minimum number of radii per seed for the fit.
            n_boot (int):
                Number of bootstrap resamples for the median CI.

        Returns:
            Dict[str, Any]:
                Dictionary with keys:
                - ok (bool): whether estimation succeeded
                - median_D (float): median of good per-seed D values (if ok)
                - median_ci (tuple): (lo, hi) bootstrap 95% CI
                - n_total_seeds (int), n_used_seeds (int)
                - r_used (List[float]) radii used for the refit
                - reason (str) when ok is False

        Example:
            >>> mf = median_filtered_D(res, total_nodes=300000, sat_thresh=0.9)
        """
        import numpy as _np
        import random as _random

        # get arrays for aggregated radii and median coverage
        box_sizes = _np.array(res.get("box_sizes", []), dtype=float)
        medians = _np.array(res.get("box_counts_median", []), dtype=float)

        # estimate total_nodes if not provided:
        if total_nodes is None:
            # collect last coverage per seed (may differ if seeds had different max radius)
            cand = []
            for s in res.get("per_seed_counts", []):
                if s:
                    cand.append(s[-1])
            if cand:
                total_nodes = int(max(cand))
            elif medians.size:
                total_nodes = int(medians[-1])
            else:
                total_nodes = 0

        # mask radii that are not saturated and are >= rmin
        non_sat_mask = medians < (sat_thresh * total_nodes)
        valid_mask = (box_sizes >= rmin) & non_sat_mask

        # if too few radii remain, return with reason
        if valid_mask.sum() < min_points:
            return {"ok": False, "reason": "few_non_saturated_points", "total_nodes": total_nodes}

        # radii that will be used for per-seed refits
        valid_rs = box_sizes[valid_mask]

        Ds = []     # store per-seed slopes
        r2s = []    # store per-seed R^2 values

        # iterate over each seed's coverage vector and fit on valid_rs
        for s in res.get("per_seed_counts", []):
            s_arr = _np.array(s, dtype=float)
            # idxs: indices in s_arr corresponding to the valid radii (r values start at 1)
            idxs = [int(r) - 1 for r in valid_rs if (int(r) - 1) < s_arr.size]
            # need at least min_points for a valid seed fit
            if len(idxs) < min_points:
                continue
            y = s_arr[idxs]  # N(r) values for this seed and selected radii
            x = _np.array([valid_rs[i] for i in range(len(idxs))], dtype=float)  # corresponding r values
            # skip invalid counts
            if _np.any(y <= 0):
                continue
            # log-log fit
            lx = _np.log(x)
            ly = _np.log(y)
            A = _np.vstack([lx, _np.ones_like(lx)]).T
            m, c = _np.linalg.lstsq(A, ly, rcond=None)[0]
            pred = m * lx + c
            ss_res = _np.sum((ly - pred) ** 2)
            ss_tot = _np.sum((ly - ly.mean()) ** 2)
            r2_local = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            Ds.append(float(m))
            r2s.append(float(r2_local))

        # filter seeds by R^2 threshold and finite slope values
        good = [d for d, rr in zip(Ds, r2s) if rr >= r2_thresh and _np.isfinite(d)]

        if not good:
            return {"ok": False, "reason": "no_good_seeds", "n_candidates": len(Ds)}

        # median and bootstrap CI for robustness
        med = float(_np.median(good))
        boots = []
        for _ in range(n_boot):
            sample = [ _random.choice(good) for _ in range(len(good)) ]
            boots.append(_np.median(sample))
        lo = float(_np.percentile(boots, 2.5))
        hi = float(_np.percentile(boots, 97.5))

        return {
            "ok": True,
            "median_D": med,
            "median_ci": (lo, hi),
            "n_total_seeds": len(res.get("per_seed_counts", [])),
            "n_used_seeds": len(good),
            "r_used": list(valid_rs),
            "total_nodes": total_nodes,
            "good_Ds_sample": good[:20],
        }

    # --- enhanced plotting: include non-saturated fit + median CI ---
    def _generate_fractal_plots(res: Dict[str, Any], out_dir: str, sample_seeds: int = 6) -> Dict[str, str]:
        """
        Generate diagnostic plots for fractal analysis.

        Plots created:
        - aggregated log-log box-counting with fit,
        - aggregated log-log excluding saturated radii and its fit (if available),
        - histogram of per-seed D with markers for median/mean and median-filtered result,
        - sample of per-seed coverage curves.

        Args:
            res (Dict[str, Any]):
                Result dictionary from calculate_fractal_dimension.
            out_dir (str):
                Directory where plots will be saved.
            sample_seeds (int):
                Number of per-seed curves to sample for the per-seed plot.

        Returns:
            Dict[str, str]:
                Mapping plot_name -> file_path for produced plots.

        Example:
            >>> paths = _generate_fractal_plots(res, "artifacts/fractal_plots")
        """

        paths: Dict[str, str] = {}
        try:
            import os
            import math
            import matplotlib.pyplot as plt
            import numpy as _np
            import random as _random
        except Exception:
            print("[WARN] matplotlib/non-core modules not available; skipping plot generation.")
            return paths

        os.makedirs(out_dir, exist_ok=True)

        # Aggregated curve (log-log) + linear fit (original)
        try:
            xs = _np.array(res.get("box_sizes", []), dtype=float)
            ys = _np.array(res.get("box_counts_median", []), dtype=float)
            if xs.size and ys.size:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.loglog(xs, ys, "o-", label="aggregated (median)")
                mask = (xs > 0) & (ys > 0)
                if mask.sum() >= 2:
                    lx = _np.log(xs[mask])
                    ly = _np.log(ys[mask])
                    coeffs = _np.polyfit(lx, ly, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    fit_y = _np.exp(intercept) * xs ** slope
                    ax.loglog(xs, fit_y, "--", label=f"fit slope={slope:.3f}")
                    ax.set_title(f"Aggregated box-counting (D={slope:.3f})")
                else:
                    ax.set_title("Aggregated box-counting (insufficient points for fit)")
                ax.set_xlabel("radius (box size)")
                ax.set_ylabel("covered nodes (median)")
                ax.grid(True, which="both", ls="--", alpha=0.4)
                ax.legend()
                p = os.path.join(out_dir, "fractal_aggregated_loglog.png")
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                plt.close(fig)
                paths["aggregated"] = p
        except Exception as e:
            print(f"[WARN] failed to plot aggregated curve: {e}")

        # Try median-filtered estimation and plot its fit (exclude saturated points)
        try:
            mf = median_filtered_D(res, total_nodes=None, sat_thresh=0.90, r2_thresh=0.8, rmin=2, min_points=3, n_boot=500)
            res["median_filtered"] = mf
            if mf.get("ok", False):
                xs = _np.array(res.get("box_sizes", []), dtype=float)
                ys = _np.array(res.get("box_counts_median", []), dtype=float)
                used_rs = _np.array(mf["r_used"], dtype=float)
                # points used for the refit
                used_mask = _np.isin(xs, used_rs)
                if used_mask.sum() >= 2:
                    lx = _np.log(xs[used_mask])
                    ly = _np.log(ys[used_mask])
                    coeffs = _np.polyfit(lx, ly, 1)
                    slope_ns = coeffs[0]
                    intercept_ns = coeffs[1]
                    fit_y = _np.exp(intercept_ns) * xs ** slope_ns

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.loglog(xs, ys, "o-", label="aggregated (median)")
                    ax.loglog(xs, fit_y, "--", label=f"nosat fit slope={slope_ns:.3f}")
                    # highlight used points
                    ax.scatter(xs[used_mask], ys[used_mask], s=80, facecolors="none", edgecolors="C1", label="used (no-sat)")
                    ax.set_title(f"Aggregated (no-sat fit) D={slope_ns:.3f}")
                    # show median filtered result box
                    med = mf.get("median_D")
                    ci = mf.get("median_ci")
                    txt = f"median_D={med:.3f}\n95% CI=({ci[0]:.3f},{ci[1]:.3f})\nused seeds={mf.get('n_used_seeds')}"
                    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, va="bottom", bbox=dict(facecolor="white", alpha=0.7))
                    ax.set_xlabel("radius (box size)")
                    ax.set_ylabel("covered nodes (median)")
                    ax.grid(True, which="both", ls="--", alpha=0.4)
                    ax.legend()
                    p = os.path.join(out_dir, "fractal_aggregated_nosat_loglog.png")
                    fig.tight_layout()
                    fig.savefig(p, dpi=150)
                    plt.close(fig)
                    paths["aggregated_nosat"] = p
        except Exception as e:
            print(f"[WARN] failed no-sat refit/plot: {e}")

        # Histogram of per-seed D (unchanged)
        try:
            vals = [v for v in res.get("per_seed_Ds", []) if _np.isfinite(v)]
            if vals:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(vals, bins=30, color="#2b8cbe", edgecolor="k", alpha=0.8)
                med = float(_np.median(vals))
                mean = float(_np.mean(vals))
                ax.axvline(med, color="orange", linestyle="--", label=f"median={med:.3f}")
                ax.axvline(mean, color="red", linestyle=":", label=f"mean={mean:.3f}")
                # if median-filtered exists, show it
                mf = res.get("median_filtered")
                if mf and mf.get("ok"):
                    ax.axvline(mf["median_D"], color="purple", linestyle="-.", label=f"median_filtered={mf['median_D']:.3f}")
                ax.set_title("Histogram of per-seed fractal dimensions (D)")
                ax.set_xlabel("D")
                ax.set_ylabel("count")
                ax.grid(axis="y", ls="--", alpha=0.3)
                ax.legend()
                p = os.path.join(out_dir, "fractal_D_histogram.png")
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                plt.close(fig)
                paths["histogram"] = p
        except Exception as e:
            print(f"[WARN] failed to plot histogram: {e}")

        # Some per-seed curves (sample few) (unchanged)
        try:
            seeds = res.get("per_seed_counts", [])
            if seeds:
                n = min(sample_seeds, len(seeds))
                idxs = _random.sample(range(len(seeds)), n)
                fig, ax = plt.subplots(figsize=(7, 5))
                for ii in idxs:
                    s = seeds[ii]
                    xs_lin = list(range(1, len(s) + 1))
                    ax.plot(xs_lin, s, "-o", alpha=0.75, label=f"seed#{ii} (pts={len(s)})")
                ax.set_xscale("linear")
                ax.set_yscale("linear")
                ax.set_xlabel("radius (box size)")
                ax.set_ylabel("covered nodes")
                ax.set_title(f"Per-seed coverage curves (sample {n})")
                ax.grid(True, ls="--", alpha=0.3)
                ax.legend(fontsize="small", loc="best")
                p = os.path.join(out_dir, "fractal_per_seed_samples.png")
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                plt.close(fig)
                paths["per_seed_samples"] = p
        except Exception as e:
            print(f"[WARN] failed to plot per-seed curves: {e}")

        return paths

    if generate_plots:
        try:
            plot_paths = _generate_fractal_plots(result, plot_dir, sample_seeds_for_plot)
            if plot_paths:
                result["plot_paths"] = plot_paths
                print(f"[INFO] Plots saved to: {plot_paths}")
            else:
                print("[INFO] No plots produced.")
        except Exception as e:
            print(f"[WARN] error generating plots: {e}")

    return result
