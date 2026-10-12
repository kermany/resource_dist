"""Population state, stored as a struct-of-arrays for vectorised updates."""

from __future__ import annotations

import numpy as np

from .config import Config


class Population:
    """Holds every per-agent array in the model.

    Attributes
    ----------
    ethnicity : (N,) int      immutable group membership
    culture   : (N, K) float  cultural trait vector in [0, 1]; dim 0 is the
                              cooperative-norm trait
    wealth    : (N,) float    current resource holdings, >= 0
    belief_group : (N, G) float  learned valence toward each ethnic group,
                              in [-1, 1]; the "stereotype" term
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        cfg.validate()
        self.cfg = cfg
        self.rng = rng
        N, G, K = cfg.n_agents, cfg.n_groups, cfg.n_culture_dims

        # --- ethnicity -------------------------------------------------
        shares = np.asarray(cfg.group_shares, dtype=float)
        shares = shares / shares.sum()
        counts = np.floor(shares * N).astype(int)
        counts[-1] += N - counts.sum()  # absorb rounding remainder
        self.ethnicity = np.repeat(np.arange(G), counts)
        rng.shuffle(self.ethnicity)
        self.group_counts = counts

        # --- culture ---------------------------------------------------
        # Group centroids are spread on a simplex-ish set of directions so
        # that separation is comparable across dimensions.
        centroids = rng.random((G, K))
        centroids = 0.5 + cfg.culture_group_separation * (centroids - 0.5) * 2.0
        self.culture_centroids = centroids
        self.culture = np.clip(
            centroids[self.ethnicity] + rng.normal(0.0, cfg.culture_sigma, size=(N, K)),
            0.0,
            1.0,
        )

        # --- wealth ----------------------------------------------------
        means = np.asarray(cfg.group_wealth_means, dtype=float)[self.ethnicity]
        # lognormal with the requested mean
        sigma = cfg.wealth_sigma
        mu = np.log(means) - 0.5 * sigma**2
        self.wealth = rng.lognormal(mu, sigma)
        self.initial_wealth = self.wealth.copy()

        # --- productivity (structural access to returns) -----------------
        gp = np.asarray(cfg.group_productivity, dtype=float)[self.ethnicity]
        ps = cfg.productivity_sigma
        self.productivity = gp * rng.lognormal(-0.5 * ps**2, ps, size=N) if ps > 0 else gp.copy()

        # --- learned beliefs -------------------------------------------
        self.belief_group = np.zeros((N, G), dtype=float)

        # --- bookkeeping -----------------------------------------------
        self.destitute_rounds = np.zeros(N, dtype=int)
        self.n_interactions = np.zeros(N, dtype=int)

    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.cfg.n_agents

    def cultural_distance(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Normalised Euclidean cultural distance in [0, 1] for pairs (i, j)."""
        d = self.culture[i] - self.culture[j]
        return np.linalg.norm(d, axis=1) / np.sqrt(self.cfg.n_culture_dims)

    def scarcity(self) -> np.ndarray:
        """How thin an agent's buffer is, in [0, 1].

        1.0 means destitute, 0.0 means holding at least `scarcity_buffer_rounds`
        rounds of subsistence in reserve.
        """
        thresh = self.cfg.subsistence * self.cfg.scarcity_buffer_rounds
        return np.clip(1.0 - self.wealth / thresh, 0.0, 1.0)
