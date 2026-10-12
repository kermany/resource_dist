"""Inequality, cooperation and segregation measures."""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Inequality
# ----------------------------------------------------------------------
def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1.0) / n)


def theil_t(x: np.ndarray) -> float:
    """Theil T index. Zero-valued observations contribute zero."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    mu = x.mean()
    if mu <= 0:
        return 0.0
    r = x / mu
    term = np.where(r > 0, r * np.log(np.where(r > 0, r, 1.0)), 0.0)
    return float(term.mean())


def theil_decomposition(x: np.ndarray, groups: np.ndarray) -> dict:
    """Split Theil T into between-group and within-group components."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    total_mu = x.mean()
    if total_mu <= 0:
        return {"total": 0.0, "between": 0.0, "within": 0.0, "between_share": 0.0}

    between = 0.0
    within = 0.0
    for g in np.unique(groups):
        m = groups == g
        xg = x[m]
        share = xg.sum() / x.sum()
        mu_g = xg.mean()
        if mu_g > 0 and share > 0:
            between += share * np.log(mu_g / total_mu)
            within += share * theil_t(xg)
    total = between + within
    return {
        "total": float(total),
        "between": float(between),
        "within": float(within),
        "between_share": float(between / total) if total > 1e-12 else 0.0,
    }


# ----------------------------------------------------------------------
# Tail concentration
# ----------------------------------------------------------------------
def top_share(x: np.ndarray, frac: float = 0.10) -> float:
    """Share of total holdings held by the richest `frac` of the population."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n_top = max(1, int(round(frac * x.size)))
    xs = np.sort(x)
    return float(xs[-n_top:].sum() / x.sum())


def bottom_share(x: np.ndarray, frac: float = 0.10) -> float:
    """Share of total holdings held by the poorest `frac` of the population."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n_bot = max(1, int(round(frac * x.size)))
    xs = np.sort(x)
    return float(xs[:n_bot].sum() / x.sum())


def tail_mean(x: np.ndarray, frac: float = 0.10, tail: str = "top") -> float:
    """Mean holding within the richest ("top") or poorest ("bottom") `frac`."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    n = max(1, int(round(frac * x.size)))
    xs = np.sort(x)
    return float(xs[-n:].mean()) if tail == "top" else float(xs[:n].mean())


def tail_group_representation(
    x: np.ndarray, groups: np.ndarray, frac: float = 0.10, tail: str = "top"
) -> dict:
    """Each group's representation in the population-wide top/bottom `frac`.

    Defined as (share of the tail belonging to the group) / (group's share of
    the population). 1.0 is proportionate; above 1.0 means the group is
    over-represented among the very richest (or poorest) agents overall.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return {}
    k = max(1, int(round(frac * n)))
    order = np.argsort(x, kind="stable")
    idx = order[-k:] if tail == "top" else order[:k]
    sel_groups = groups[idx]
    out = {}
    for g in np.unique(groups):
        pop_share = float((groups == g).mean())
        occ_share = float((sel_groups == g).mean())
        out[int(g)] = occ_share / pop_share if pop_share > 0 else 0.0
    return out


# ----------------------------------------------------------------------
# Segregation
# ----------------------------------------------------------------------
def categorical_assortativity(edges: np.ndarray, attr: np.ndarray) -> float:
    """Newman's assortativity coefficient for a categorical node attribute."""
    if edges.size == 0:
        return 0.0
    a_end = attr[edges[:, 0]]
    b_end = attr[edges[:, 1]]
    cats = np.unique(attr)
    k = len(cats)
    idx = {c: n for n, c in enumerate(cats)}
    e = np.zeros((k, k))
    for u, v in zip(a_end, b_end):
        # count both orientations: the graph is undirected
        e[idx[u], idx[v]] += 1
        e[idx[v], idx[u]] += 1
    e /= e.sum()
    a = e.sum(axis=1)
    tr = np.trace(e)
    denom = 1.0 - (a * a).sum()
    return float((tr - (a * a).sum()) / denom) if denom > 1e-12 else 0.0


def cross_group_tie_share(edges: np.ndarray, attr: np.ndarray) -> float:
    if edges.size == 0:
        return 0.0
    return float((attr[edges[:, 0]] != attr[edges[:, 1]]).mean())


# ----------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------
def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, computed without scipy."""
    if len(a) < 2:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------
def snapshot(pop, net, round_stats: dict, t: int) -> dict:
    """Assemble one row of the time series."""
    w = pop.wealth
    eth = pop.ethnicity
    edges = net.edge_array()
    dec = theil_decomposition(w, eth)

    row = {
        "t": t,
        "gini": gini(w),
        "theil_total": dec["total"],
        "theil_between": dec["between"],
        "theil_within": dec["within"],
        "between_share": dec["between_share"],
        "mean_wealth": float(w.mean()),
        "median_wealth": float(np.median(w)),
        "destitute_frac": float((w < 1e-6).mean()),
        "assortativity": categorical_assortativity(edges, eth),
        "cross_tie_share": cross_group_tie_share(edges, eth),
        "mean_degree": float(net.degrees().mean()),
        "rank_persistence": rank_correlation(pop.initial_wealth, w),
        # tails, population-wide
        "top1_share": top_share(w, 0.01),
        "top10_share": top_share(w, 0.10),
        "bottom10_share": bottom_share(w, 0.10),
        "top1_mean": tail_mean(w, 0.01, "top"),
        "top10_mean": tail_mean(w, 0.10, "top"),
        "bottom10_mean": tail_mean(w, 0.10, "bottom"),
    }
    row.update(round_stats)

    top1_repr = tail_group_representation(w, eth, 0.01, "top")
    top10_repr = tail_group_representation(w, eth, 0.10, "top")
    bottom10_repr = tail_group_representation(w, eth, 0.10, "bottom")

    for g in range(pop.cfg.n_groups):
        m = eth == g
        wg = w[m]
        row[f"wealth_g{g}"] = float(wg.mean()) if m.any() else 0.0
        row[f"share_g{g}"] = float(wg.sum() / w.sum()) if w.sum() > 0 else 0.0
        row[f"gini_g{g}"] = gini(wg)
        row[f"destitute_g{g}"] = float((wg < 1e-6).mean()) if m.any() else 0.0
        # within-group concentration: this group's own top/bottom tail as a
        # share of this group's own total wealth
        row[f"top1_share_g{g}"] = top_share(wg, 0.01)
        row[f"top10_share_g{g}"] = top_share(wg, 0.10)
        row[f"bottom10_share_g{g}"] = bottom_share(wg, 0.10)
        row[f"top10_mean_g{g}"] = tail_mean(wg, 0.10, "top")
        row[f"bottom10_mean_g{g}"] = tail_mean(wg, 0.10, "bottom")
        # this group's representation among the population-wide top/bottom
        # tail, relative to its population share (1.0 = proportionate)
        row[f"top1_repr_g{g}"] = top1_repr.get(g, 0.0)
        row[f"top10_repr_g{g}"] = top10_repr.get(g, 0.0)
        row[f"bottom10_repr_g{g}"] = bottom10_repr.get(g, 0.0)
        # mean stereotype this group holds about out-groups vs its own
        bg = pop.belief_group[m]
        if m.any():
            row[f"belief_in_g{g}"] = float(bg[:, g].mean())
            out = [h for h in range(pop.cfg.n_groups) if h != g]
            row[f"belief_out_g{g}"] = float(bg[:, out].mean()) if out else 0.0
    return row
