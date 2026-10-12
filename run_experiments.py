"""Baseline scenarios and parameter sweeps.

Writes outputs/scenarios.json and outputs/sweeps.json, which the analysis
figures and the HTML viewer both read.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from resdist import Config, Simulation

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

N_SEEDS = 5
SCEN = dict(n_agents=500, n_rounds=350, record_every=5)
SWEEP = dict(n_agents=300, n_rounds=250, record_every=25)

METRICS = [
    "gini", "between_share", "theil_between", "theil_within", "coop_rate",
    "coop_in", "coop_out", "ingroup_bias", "destitute_frac", "assortativity",
    "cross_tie_share", "mean_wealth", "rank_persistence", "rank_persistence_recent",
    "top1_share", "top10_share", "bottom10_share",
]

#: Per-group series/finals tracked alongside the population-wide METRICS above.
GROUP_KEYS = [
    "wealth_g", "share_g", "destitute_g",
    "top1_share_g", "top10_share_g", "bottom10_share_g",
    "top1_repr_g", "top10_repr_g", "bottom10_repr_g",
]


# ----------------------------------------------------------------------
def scenarios() -> dict[str, Config]:
    base = Config(label="baseline", **SCEN)
    return {
        "baseline": base,
        # --- switch off one channel at a time -------------------------
        "no_taste_bias": base.variant(
            label="no_taste_bias", beta_ethnic=0.0),
        "no_ethnic_homophily": base.variant(
            label="no_ethnic_homophily", homophily_ethnic=0.0),
        "no_stereotypes": base.variant(
            label="no_stereotypes", lambda_group=0.0),
        "no_desperation": base.variant(
            label="no_desperation", beta_scarcity=0.0),
        # --- structural advantage -------------------------------------
        "equal_access": base.variant(
            label="equal_access", group_productivity=(1.0, 1.0, 1.0),
            group_wealth_means=(10.0, 10.0, 10.0)),
        # isolates whether identity bias *alone* can open a group gap
        "equal_access_no_bias": base.variant(
            label="equal_access_no_bias", group_productivity=(1.0, 1.0, 1.0),
            group_wealth_means=(10.0, 10.0, 10.0),
            beta_ethnic=0.0, homophily_ethnic=0.0),
        # --- policy levers --------------------------------------------
        "redistribution": base.variant(
            label="redistribution", tax_rate=0.35),
        "forced_contact": base.variant(
            label="forced_contact", p_local=0.20, rewire_rate=0.0),
        # --- equal starting wealth / asymmetric in-group bias ----------
        # Baseline productivity stays unequal: the paper's finding is that
        # persistent between-group inequality is sustained by differential
        # access to returns, not by the initial stock, so this isolates
        # whether starting level alone matters.
        "equal_start": base.variant(
            label="equal_start", group_wealth_means=(10.0, 10.0, 10.0)),
        # Group C (the smallest, poorest, least productive group) gets a
        # doubled co-ethnic preference relative to A and B -- e.g. a minority
        # that has developed stronger in-group solidarity. Starting wealth
        # and productivity stay at baseline (unequal).
        "elevated_minority_bias": base.variant(
            label="elevated_minority_bias", group_beta_ethnic=(1.2, 1.2, 2.4)),
        # The same asymmetric bias, but agents start with equal wealth across
        # groups -- isolates whether the elevated in-group bias can open a
        # gap on its own, absent a starting head start.
        "elevated_minority_bias_equal_start": base.variant(
            label="elevated_minority_bias_equal_start",
            group_beta_ethnic=(1.2, 1.2, 2.4),
            group_wealth_means=(10.0, 10.0, 10.0)),
        # --- clean room: population size is the *only* structural difference --
        # No productivity gap, no starting-wealth gap, no cultural-centroid
        # gap. Isolates what group size alone -- with or without an
        # elevated in-group bias for the minority -- can do by itself.
        "size_only": base.variant(
            label="size_only", group_wealth_means=(10.0, 10.0, 10.0),
            group_productivity=(1.0, 1.0, 1.0), culture_group_separation=0.0),
        "size_only_elevated_minority_bias": base.variant(
            label="size_only_elevated_minority_bias",
            group_wealth_means=(10.0, 10.0, 10.0),
            group_productivity=(1.0, 1.0, 1.0), culture_group_separation=0.0,
            group_beta_ethnic=(1.2, 1.2, 2.4)),
    }


def _run_one(args):
    cfg_kwargs, seed = args
    cfg = Config(**cfg_kwargs).variant(seed=seed)
    sim = Simulation(cfg)
    hist = sim.run()
    return {
        "history": hist,
        "coop_matrix": np.nan_to_num(sim.cooperation_matrix()).tolist(),
        "final_wealth": sim.pop.wealth.tolist(),
        "ethnicity": sim.pop.ethnicity.tolist(),
        "belief_group": sim.pop.belief_group.mean(axis=0).tolist(),
        "edges": sim.net.edge_array().tolist(),
    }


def aggregate(runs: list[dict]) -> dict:
    """Mean and 10-90 band of each metric across seeds, aligned on t."""
    ts = [r["t"] for r in runs[0]["history"]]
    series = {"t": ts}
    for m in METRICS:
        arr = np.array([[h.get(m, np.nan) for h in r["history"]] for r in runs], float)
        with np.errstate(invalid="ignore"):
            series[m] = np.nanmean(arr, axis=0).tolist()
            series[m + "_lo"] = np.nanpercentile(arr, 10, axis=0).tolist()
            series[m + "_hi"] = np.nanpercentile(arr, 90, axis=0).tolist()
    for g in range(3):
        for pre in GROUP_KEYS + ["belief_in_g", "belief_out_g"]:
            key = f"{pre}{g}"
            arr = np.array([[h.get(key, np.nan) for h in r["history"]] for r in runs], float)
            series[key] = np.nanmean(arr, axis=0).tolist()

    final = {}
    for m in METRICS:
        vals = [np.nanmean([h[m] for h in r["history"][-8:]]) for r in runs]
        final[m] = float(np.mean(vals))
        final[m + "_sd"] = float(np.std(vals))
    for g in range(3):
        for pre in GROUP_KEYS:
            key = f"{pre}{g}"
            final[key] = float(np.mean([r["history"][-1][key] for r in runs]))

    coop = np.mean([r["coop_matrix"] for r in runs], axis=0).tolist()
    return {"series": series, "final": final, "coop_matrix": coop}


# ----------------------------------------------------------------------
def run_scenarios(pool) -> dict:
    scen = scenarios()
    jobs, index = [], []
    for name, cfg in scen.items():
        for s in range(N_SEEDS):
            jobs.append((asdict(cfg), s))
            index.append(name)
    print(f"scenarios: {len(jobs)} runs")
    results = pool.map(_run_one, jobs)

    out = {}
    for name, cfg in scen.items():
        runs = [r for r, n in zip(results, index) if n == name]
        agg = aggregate(runs)
        agg["config"] = asdict(cfg)
        # wealth distribution by group from the first seed, for the viewer
        r0 = runs[0]
        agg["wealth_sample"] = {
            "wealth": [round(w, 3) for w in r0["final_wealth"]],
            "ethnicity": r0["ethnicity"],
        }
        agg["belief_group"] = np.mean([r["belief_group"] for r in runs], axis=0).tolist()
        out[name] = agg
    return out


def run_sweeps(pool) -> dict:
    base = Config(**SWEEP)
    sweeps = {}

    # --- A: identity bias x stereotype learning -----------------------
    xs = [0.0, 0.6, 1.2, 1.8, 2.4]
    ys = [0.0, 0.75, 1.5, 2.25]
    jobs, index = [], []
    for x in xs:
        for y in ys:
            cfg = base.variant(beta_ethnic=x, lambda_group=y,
                               label=f"eth{x}_lam{y}")
            for s in range(3):
                jobs.append((asdict(cfg), s))
                index.append((x, y))
    print(f"sweep A: {len(jobs)} runs")
    res = pool.map(_run_one, jobs)
    sweeps["bias_x_learning"] = {
        "x_name": "beta_ethnic", "y_name": "lambda_group",
        "x": xs, "y": ys,
        "cells": _cells(res, index, xs, ys),
    }

    # --- B: desperation x exposure to crisis --------------------------
    xs = [0.0, -1.5, -3.0, -4.5]
    ys = [0.0, 0.02, 0.05, 0.09]
    jobs, index = [], []
    for x in xs:
        for y in ys:
            cfg = base.variant(beta_scarcity=x, crisis_prob=y,
                               label=f"sca{x}_cri{y}")
            for s in range(3):
                jobs.append((asdict(cfg), s))
                index.append((x, y))
    print(f"sweep B: {len(jobs)} runs")
    res = pool.map(_run_one, jobs)
    sweeps["desperation_x_crisis"] = {
        "x_name": "beta_scarcity", "y_name": "crisis_prob",
        "x": xs, "y": ys,
        "cells": _cells(res, index, xs, ys),
    }
    return sweeps


def _cells(results, index, xs, ys) -> list:
    cells = []
    for x in xs:
        for y in ys:
            runs = [r for r, k in zip(results, index) if k == (x, y)]
            vals = {}
            for m in METRICS:
                v = [np.nanmean([h[m] for h in r["history"][-4:]]) for r in runs]
                vals[m] = float(np.mean(v))
                vals[m + "_sd"] = float(np.std(v))
            cells.append({"x": x, "y": y, **vals})
    return cells


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    with Pool(2) as pool:
        scen = run_scenarios(pool)
        (OUT / "scenarios.json").write_text(json.dumps(scen))
        print(f"  wrote scenarios.json  ({time.time()-t0:.0f}s)")

        sw = run_sweeps(pool)
        (OUT / "sweeps.json").write_text(json.dumps(sw))
        print(f"  wrote sweeps.json  ({time.time()-t0:.0f}s)")

    print("\nfinal metrics by scenario")
    hdr = f"{'scenario':<16}{'gini':>7}{'btwn%':>7}{'coop':>7}{'bias':>7}{'dest%':>7}{'assort':>8}{'rankp':>7}{'rankp100':>9}"
    print(hdr)
    for name, d in scen.items():
        f = d["final"]
        print(f"{name:<16}{f['gini']:7.3f}{f['between_share']:7.3f}{f['coop_rate']:7.3f}"
              f"{f['ingroup_bias']:7.3f}{f['destitute_frac']:7.3f}{f['assortativity']:8.3f}"
              f"{f['rank_persistence']:7.3f}{f['rank_persistence_recent']:9.3f}")


if __name__ == "__main__":
    main()
