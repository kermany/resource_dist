"""Main simulation loop."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from . import games, metrics
from .config import Config
from .decision import DecisionSystem
from .network import SocialNetwork
from .population import Population


class Simulation:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.pop = Population(cfg, self.rng)
        self.net = SocialNetwork(self.pop, cfg, self.rng)
        self.dec = DecisionSystem(cfg, self.pop)
        self.history: list[dict] = []
        # rolling buffer of past wealth vectors for the short-run mobility measure
        self._wealth_lag: deque = deque(maxlen=max(1, cfg.mobility_lag) + 1)
        self._wealth_lag.append(self.pop.wealth.copy())
        self.coop_matrix = np.zeros((cfg.n_groups, cfg.n_groups))
        self.coop_counts = np.zeros((cfg.n_groups, cfg.n_groups))

    # ------------------------------------------------------------------
    def _economy(self) -> None:
        cfg, pop, rng = self.cfg, self.pop, self.rng
        w = np.maximum(pop.wealth, 0.0)

        income = pop.productivity * (
            cfg.base_income + cfg.capital_return * np.power(w, cfg.capital_exponent)
        )
        if cfg.income_shock_sigma > 0:
            s = cfg.income_shock_sigma
            income *= rng.lognormal(-0.5 * s**2, s, size=len(w))

        if cfg.tax_rate > 0:
            tax = cfg.tax_rate * income
            income = income - tax + tax.sum() / len(w)

        w = w + income

        if cfg.crisis_prob > 0:
            hit = rng.random(len(w)) < cfg.crisis_prob
            w = np.where(hit, w * (1.0 - cfg.crisis_severity), w)
            self._crises_this_round = int(hit.sum())
        pop.wealth = w

    def _settle(self) -> None:
        cfg, pop = self.cfg, self.pop
        pop.wealth = pop.wealth - cfg.subsistence
        pop.wealth *= 1.0 - cfg.depreciation
        pop.destitute_rounds += (pop.wealth < 1e-6).astype(int)
        pop.wealth = np.clip(pop.wealth, 0.0, None)

    # ------------------------------------------------------------------
    def step(self, t: int) -> dict:
        cfg, pop, net, dec, rng = self.cfg, self.pop, self.net, self.dec, self.rng

        self._economy()

        i, j = net.match(rng)
        if len(i) == 0:
            self._settle()
            return {}

        p_i = dec.p_cooperate(i, j, net, rng)   # i's willingness toward j
        p_j = dec.p_cooperate(j, i, net, rng)   # j's willingness toward i

        is_trust = rng.random(len(i)) < cfg.trust_game_frac
        delta = np.zeros_like(pop.wealth)
        sig_i = np.zeros(len(i))
        sig_j = np.zeros(len(i))
        coop_i = np.zeros(len(i))
        coop_j = np.zeros(len(i))

        for mask, fn in ((~is_trust, games.play_donation), (is_trust, games.play_trust)):
            if not mask.any():
                continue
            out = fn(cfg, pop.wealth, i[mask], j[mask], p_i[mask], p_j[mask], rng)
            delta += out.delta
            sig_i[mask] = out.sig_i
            sig_j[mask] = out.sig_j
            coop_i[mask] = out.coop_i
            coop_j[mask] = out.coop_j

        pop.wealth = pop.wealth + delta
        self._settle()

        # --- learning ---------------------------------------------------
        dec.update(i, j, sig_i)
        dec.update(j, i, sig_j)
        dec.decay_dyads(rate=0.005)
        pop.n_interactions[i] += 1
        pop.n_interactions[j] += 1

        # --- cultural transmission -------------------------------------
        if cfg.culture_drift > 0:
            mutual = (coop_i > 0.5) & (coop_j > 0.5)
            if mutual.any():
                a, b = i[mutual], j[mutual]
                ca, cb = pop.culture[a].copy(), pop.culture[b].copy()
                pop.culture[a] = np.clip(ca + cfg.culture_drift * (cb - ca), 0, 1)
                pop.culture[b] = np.clip(cb + cfg.culture_drift * (ca - cb), 0, 1)

        # --- tie revision ----------------------------------------------
        signal_map: dict[int, dict[int, float]] = defaultdict(dict)
        for a, b, sa, sb in zip(i.tolist(), j.tolist(), sig_i.tolist(), sig_j.tolist()):
            signal_map[a][b] = sa
            signal_map[b][a] = sb
        net.rewire(pop, signal_map, rng)

        # --- round statistics -------------------------------------------
        same = pop.ethnicity[i] == pop.ethnicity[j]
        allcoop = np.concatenate([coop_i, coop_j])
        allsame = np.concatenate([same, same])
        in_c = float(allcoop[allsame].mean()) if allsame.any() else np.nan
        out_c = float(allcoop[~allsame].mean()) if (~allsame).any() else np.nan

        gi, gj = pop.ethnicity[i], pop.ethnicity[j]
        np.add.at(self.coop_matrix, (gi, gj), coop_i)
        np.add.at(self.coop_counts, (gi, gj), 1.0)
        np.add.at(self.coop_matrix, (gj, gi), coop_j)
        np.add.at(self.coop_counts, (gj, gi), 1.0)

        return {
            "coop_rate": float(allcoop.mean()),
            "coop_in": in_c,
            "coop_out": out_c,
            "ingroup_bias": (in_c - out_c) if not (np.isnan(in_c) or np.isnan(out_c)) else np.nan,
            "n_interactions": len(i),
        }

    # ------------------------------------------------------------------
    def run(self, progress: bool = False) -> list[dict]:
        cfg = self.cfg
        recent: list[dict] = []
        for t in range(1, cfg.n_rounds + 1):
            stats = self.step(t)
            recent.append(stats)
            self._wealth_lag.append(self.pop.wealth.copy())
            if t % cfg.record_every == 0 or t == cfg.n_rounds:
                # average the round stats since the last record for smoothness
                keys = [k for k in recent[-1].keys()] if recent[-1] else []
                agg = {
                    k: float(np.nanmean([r[k] for r in recent if k in r]))
                    for k in keys
                }
                agg["rank_persistence_recent"] = metrics.rank_correlation(
                    self._wealth_lag[0], self.pop.wealth)
                self.history.append(metrics.snapshot(self.pop, self.net, agg, t))
                recent = []
            if progress and t % max(1, cfg.n_rounds // 10) == 0:
                print(f"  [{cfg.label}] round {t}/{cfg.n_rounds}", flush=True)
        return self.history

    # ------------------------------------------------------------------
    def cooperation_matrix(self) -> np.ndarray:
        """Mean cooperation intensity of row-group toward column-group."""
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(self.coop_counts > 0, self.coop_matrix / self.coop_counts, np.nan)
        return m


def run_config(cfg: Config, progress: bool = False):
    sim = Simulation(cfg)
    sim.run(progress=progress)
    return sim
