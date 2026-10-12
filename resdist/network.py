"""Homophilous social network: construction, endogenous rewiring, matching."""

from __future__ import annotations

import numpy as np

from .config import Config
from .population import Population


class SocialNetwork:
    """Undirected graph over agents, stored as adjacency sets.

    Ties form preferentially between agents who share ethnicity, are
    culturally close, and hold similar amounts of resources. Ties are revised
    over time in response to how partners behaved, which lets segregation
    emerge (or dissolve) endogenously rather than being fixed at t=0.
    """

    def __init__(self, pop: Population, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.pop = pop
        self.adj: list[set[int]] = [set() for _ in range(pop.n)]
        self._build(pop, cfg, rng)

    # ------------------------------------------------------------------
    def _tie_logits(self, pop: Population, cfg: Config, rows: np.ndarray) -> np.ndarray:
        """Unnormalised log-affinity of `rows` toward every agent."""
        eth = pop.ethnicity
        diff_eth = (eth[rows][:, None] != eth[None, :]).astype(float)

        cdiff = pop.culture[rows][:, None, :] - pop.culture[None, :, :]
        cdist = np.linalg.norm(cdiff, axis=2) / np.sqrt(cfg.n_culture_dims)

        lw = np.log1p(np.maximum(pop.wealth, 0.0))
        wdist = np.abs(lw[rows][:, None] - lw[None, :])
        wscale = lw.std() + 1e-9

        return -(
            cfg.homophily_ethnic * diff_eth
            + cfg.homophily_cultural * cdist
            + cfg.homophily_wealth * (wdist / wscale)
        )

    def _build(self, pop: Population, cfg: Config, rng: np.random.Generator) -> None:
        N = pop.n
        block = 256
        target_edges = int(round(cfg.mean_degree * N / 2))

        # Two passes: first collect logits blockwise to find the scaling that
        # yields the requested mean degree, then sample.
        all_p = np.empty((N, N), dtype=np.float32)
        for start in range(0, N, block):
            rows = np.arange(start, min(start + block, N))
            logits = self._tie_logits(pop, cfg, rows)
            all_p[rows] = np.exp(logits).astype(np.float32)
        np.fill_diagonal(all_p, 0.0)
        all_p = np.triu(all_p, k=1)

        total = all_p.sum()
        if total <= 0:
            return
        # Scale affinities into probabilities with the right expected count.
        scale = target_edges / total
        probs = np.clip(all_p * scale, 0.0, 1.0)
        draws = rng.random(probs.shape).astype(np.float32) < probs
        ii, jj = np.nonzero(draws)
        for a, b in zip(ii.tolist(), jj.tolist()):
            self.adj[a].add(b)
            self.adj[b].add(a)

    # ------------------------------------------------------------------
    def degrees(self) -> np.ndarray:
        return np.array([len(s) for s in self.adj], dtype=int)

    def edge_array(self) -> np.ndarray:
        edges = [(a, b) for a in range(len(self.adj)) for b in self.adj[a] if b > a]
        return np.array(edges, dtype=int).reshape(-1, 2)

    def social_proximity(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Tie indicator plus Jaccard overlap of neighbourhoods, in [0, 2]."""
        out = np.zeros(len(i), dtype=float)
        adj = self.adj
        for k, (a, b) in enumerate(zip(i.tolist(), j.tolist())):
            na, nb = adj[a], adj[b]
            tie = 1.0 if b in na else 0.0
            union = len(na | nb)
            jac = (len(na & nb) / union) if union else 0.0
            out[k] = tie + jac
        return out

    # ------------------------------------------------------------------
    def match(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Pair agents into a matching for one round of interaction.

        With probability `p_local` an agent seeks a partner among its unmatched
        network neighbours; otherwise it draws from the unmatched population.
        """
        N = len(self.adj)
        order = rng.permutation(N)
        matched = np.zeros(N, dtype=bool)
        free = list(order)
        free_set = set(free)
        left, right = [], []

        p_local = self.cfg.p_local
        for a in order:
            a = int(a)
            if matched[a]:
                continue
            free_set.discard(a)
            partner = -1
            if p_local > 0 and rng.random() < p_local:
                cands = [c for c in self.adj[a] if not matched[c]]
                if cands:
                    partner = int(cands[rng.integers(len(cands))])
            if partner < 0:
                if not free_set:
                    break
                # sample from the unmatched pool
                pool = free_set
                idx = rng.integers(len(pool))
                for n_, c in enumerate(pool):
                    if n_ == idx:
                        partner = int(c)
                        break
            if partner < 0:
                break
            matched[a] = True
            matched[partner] = True
            free_set.discard(partner)
            left.append(a)
            right.append(partner)

        return np.array(left, dtype=int), np.array(right, dtype=int)

    # ------------------------------------------------------------------
    def rewire(self, pop: Population, payoff_signal: dict, rng: np.random.Generator) -> None:
        """Let a fraction of agents drop a disappointing tie and form a new one.

        `payoff_signal` maps agent -> {alter: signal} accumulated this round;
        agents preferentially cut ties to alters who defected on them and
        preferentially attach to similar others.
        """
        cfg = self.cfg
        N = len(self.adj)
        n_rewire = rng.binomial(N, cfg.rewire_rate)
        if n_rewire == 0:
            return
        movers = rng.choice(N, size=n_rewire, replace=False)

        for a in movers.tolist():
            nbrs = self.adj[a]
            if not nbrs:
                continue
            # --- drop: prefer the alter with the worst recent signal --------
            sig = payoff_signal.get(a, {})
            worst, worst_val = None, np.inf
            for c in nbrs:
                v = sig.get(c, 0.0)
                if v < worst_val:
                    worst, worst_val = c, v
            if worst is None:
                continue
            self.adj[a].discard(worst)
            self.adj[worst].discard(a)

            # --- add: sample candidates and pick by homophilous affinity ----
            cands = rng.integers(0, N, size=12)
            cands = [int(c) for c in cands if c != a and c not in self.adj[a]]
            if not cands:
                continue
            cand = np.array(cands)
            same_eth = (pop.ethnicity[cand] == pop.ethnicity[a]).astype(float)
            cdist = pop.cultural_distance(np.full(len(cand), a), cand)
            lw = np.log1p(np.maximum(pop.wealth, 0.0))
            wdist = np.abs(lw[cand] - lw[a]) / (lw.std() + 1e-9)
            logit = (
                cfg.homophily_ethnic * same_eth
                - cfg.homophily_cultural * cdist
                - cfg.homophily_wealth * wdist
            )
            w = np.exp(logit - logit.max())
            w = w / w.sum()
            pick = int(cand[rng.choice(len(cand), p=w)])
            self.adj[a].add(pick)
            self.adj[pick].add(a)
