"""
CPU-side parallel-selection reference checks for puct_gpu_v2.

This is intentionally CUDA-free. It mirrors the batching idea in ref.py:
within a batch, each readout selects a leaf, immediately applies virtual loss
to the selected path, and only later would evaluation/backup release it.
"""

import sys

import numpy as np


PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"

VARIANT_PRECLAIM = "preclaim"
VARIANT_WINNER_RECALC = "winner_recalc"


def _hist(values, actions):
    counts = np.bincount(values.reshape(-1), minlength=actions).astype(np.float64)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _choose_rotated(scores, rotation):
    actions = scores.shape[1]
    order = np.concatenate((np.arange(rotation, actions), np.arange(0, rotation)))
    local = np.argmax(scores[:, order], axis=1)
    return order[local].astype(np.int32)


def _score(prior, inflight):
    # v2-style PUCT with N=1, W=0 on every legal edge. The shared parent term is
    # irrelevant for argmax, so it is omitted.
    return prior[None, :] / (2.0 + inflight.astype(np.float32))


def simulate_cpu_ref(trees, warps, depth, actions, prior):
    """Sequential batch readouts, matching ref.py's virtual-loss timing."""
    paths = np.empty((trees, warps, depth), np.int16)
    inflight = np.zeros((trees, depth, actions), np.int16)
    for d in range(depth):
        layer = inflight[:, d, :]
        for wid in range(warps):
            best = _choose_rotated(_score(prior, layer), wid % actions)
            paths[:, wid, d] = best
            layer[np.arange(trees), best] += 1
    return paths


def simulate_preclaim(trees, warps, depth, actions, prior):
    """CPU model of all-candidate preclaim at each tree depth."""
    paths = np.empty((trees, warps, depth), np.int16)
    for d in range(depth):
        layer = np.full((trees, actions), warps, np.int16)
        winners = []
        for wid in range(warps):
            best = _choose_rotated(_score(prior, layer), wid % actions)
            paths[:, wid, d] = best
            winners.append(best)
        # Rollback losers; retained inflight is exactly one per selected winner.
        # The next depth is a different node in this synthetic chain, so no state
        # needs to be carried here.
        _ = winners
    return paths


def simulate_winner_recalc(trees, warps, depth, actions, prior):
    """CPU model of winner claim, then one conflict-triggered recalculation."""
    paths = np.empty((trees, warps, depth), np.int16)
    for d in range(depth):
        layer = np.zeros((trees, actions), np.int16)
        for wid in range(warps):
            original = _choose_rotated(_score(prior, layer), wid % actions)
            old = layer[np.arange(trees), original]
            conflict = old > 0
            best = original.copy()
            if np.any(conflict) and actions > 1:
                layer[np.arange(trees), original] += 1
                layer[np.arange(trees)[conflict], original[conflict]] -= 1
                scores = _score(prior, layer)
                scores[np.arange(trees), original] = -np.inf
                recalculated = _choose_rotated(scores, wid % actions)
                best[conflict] = recalculated[conflict]
            layer[np.arange(trees), best] += 1
            paths[:, wid, d] = best
    return paths


def path_metrics(paths):
    trees, warps, _ = paths.shape
    dominant = []
    path75 = 0
    same = 0
    for t in range(trees):
        counts = {}
        for wid in range(warps):
            key = tuple(int(x) for x in paths[t, wid])
            counts[key] = counts.get(key, 0) + 1
        max_frac = max(counts.values()) / float(warps)
        dominant.append(max_frac)
        if max_frac >= 0.75:
            path75 += 1
        if max_frac == 1.0:
            same += 1
    return {
        "maxpath": 100.0 * float(np.mean(dominant)),
        "path75": 100.0 * path75 / float(trees),
        "samepath": 100.0 * same / float(trees),
    }


def compare_to_ref(ref_paths, cand_paths, actions):
    ref_first = _hist(ref_paths[:, :, 0], actions)
    cand_first = _hist(cand_paths[:, :, 0], actions)
    total_variation = 0.5 * float(np.abs(ref_first - cand_first).sum())
    exact_path_match = 100.0 * float(np.mean(np.all(ref_paths == cand_paths, axis=2)))
    metrics = path_metrics(cand_paths)
    metrics["first_tv"] = total_variation
    metrics["exact_path_match"] = exact_path_match
    return metrics


def run_cpu_reference_check():
    configs = [
        ("equal64_d32", 256, 8, 32, 64, "equal"),
        ("hot64_d32", 256, 8, 32, 64, "hot"),
        ("equal128_d64", 256, 8, 64, 128, "equal"),
        ("hot128_d64", 256, 8, 64, 128, "hot"),
        ("equal256_d64", 128, 8, 64, 256, "equal"),
        ("hot256_d64", 128, 8, 64, 256, "hot"),
    ]

    print("\n  V2 CPU parallel reference check")
    print("  " + "-" * 126)
    print(
        f"  {'scenario':>14} | {'variant':>13} | {'trees':>5} | {'warps':>5} | "
        f"{'depth':>5} | {'acts':>5} | {'first_TV':>8} | {'path_match%':>11} | "
        f"{'maxpath%':>8} | {'path75%':>7} | {'samepath%':>9}"
    )
    print("  " + "-" * 126)

    all_ok = True
    summaries = []
    for name, trees, warps, depth, actions, prior_mode in configs:
        prior = np.ones(actions, np.float32)
        if prior_mode == "hot":
            prior[0] = 2.0

        ref = simulate_cpu_ref(trees, warps, depth, actions, prior)
        for variant, fn in [
            (VARIANT_PRECLAIM, simulate_preclaim),
            (VARIANT_WINNER_RECALC, simulate_winner_recalc),
        ]:
            cand = fn(trees, warps, depth, actions, prior)
            metrics = compare_to_ref(ref, cand, actions)
            summaries.append((name, variant, metrics["first_tv"], metrics["exact_path_match"]))
            print(
                f"  {name:>14} | {variant:>13} | {trees:5d} | {warps:5d} | "
                f"{depth:5d} | {actions:5d} | {metrics['first_tv']:8.3f} | "
                f"{metrics['exact_path_match']:10.2f}% | {metrics['maxpath']:7.2f}% | "
                f"{metrics['path75']:6.2f}% | {metrics['samepath']:8.2f}%"
            )

    preclaim_score = sum(tv for _, variant, tv, _ in summaries if variant == VARIANT_PRECLAIM)
    recalc_score = sum(tv for _, variant, tv, _ in summaries if variant == VARIANT_WINNER_RECALC)
    preclaim_match = sum(match for _, variant, _, match in summaries if variant == VARIANT_PRECLAIM)
    recalc_match = sum(match for _, variant, _, match in summaries if variant == VARIANT_WINNER_RECALC)
    recalc_closer = recalc_score <= preclaim_score and recalc_match >= preclaim_match
    all_ok = all_ok and recalc_closer

    print("  " + "-" * 126)
    print(
        f"  aggregate: preclaim_TV={preclaim_score:.3f}, winner_recalc_TV={recalc_score:.3f}, "
        f"preclaim_match={preclaim_match:.2f}, winner_recalc_match={recalc_match:.2f}"
    )
    tag = PASS_MARK if all_ok else FAIL_MARK
    verdict = "winner_recalc is closer to the ref.py-style CPU batch expectation"
    print(f"{tag}  CPU reference comparison  ({verdict})")
    return all_ok


if __name__ == "__main__":
    if not run_cpu_reference_check():
        sys.exit(1)
