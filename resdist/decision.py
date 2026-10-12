"""The hybrid cooperation rule.

A structural logistic baseline sets the prior propensity to cooperate from
identity, culture, social proximity and economic position. Two learned terms
sit on top of it: a slow-moving valence toward each *ethnic group* (the
stereotype channel, which can become self-fulfilling) and a fast-moving
valence toward the *specific partner* (direct reciprocity).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import Config
from .network import SocialNetwork
from .population import Population


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class DecisionSystem:
    def __init__(self, cfg: Config, pop: Population):
        self.cfg = cfg
        self.pop = pop
        # sparse dyadic memory: dyad[i][j] in [-1, 1]
        self.dyad: dict[int, dict[int, float]] = defaultdict(dict)

        # The ethnic term is centred on the chance that two agents drawn at
        # random share an ethnicity. Without this, raising beta_ethnic also
        # raises the average propensity to cooperate, so a "no ethnic bias"
        # counterfactual would silently also be a "less cooperation overall"
        # counterfactual and the two effects could not be told apart.
        shares = np.asarray(cfg.group_shares, dtype=float)
        shares = shares / shares.sum()
        self.p_same_ethnicity = float((shares**2).sum())

        # Per-actor co-ethnic preference. Usually uniform, but a group can be
        # given an elevated (or suppressed) taste for its own via
        # `group_beta_ethnic` without touching the centring constant above,
        # which depends only on group sizes.
        if cfg.group_beta_ethnic is not None:
            self.beta_ethnic_by_group = np.asarray(cfg.group_beta_ethnic, dtype=float)
        else:
            self.beta_ethnic_by_group = np.full(cfg.n_groups, cfg.beta_ethnic, dtype=float)

    # ------------------------------------------------------------------
    def components(
        self, i: np.ndarray, j: np.ndarray, net: SocialNetwork
    ) -> dict[str, np.ndarray]:
        """Return each additive term of the latent propensity of i toward j."""
        cfg, pop = self.cfg, self.pop
        w = pop.wealth
        w_ref = max(w.mean(), 1e-9)

        same_eth = (pop.ethnicity[i] == pop.ethnicity[j]).astype(float)
        beta_eth_i = self.beta_ethnic_by_group[pop.ethnicity[i]]
        cult_sim = 1.0 - pop.cultural_distance(i, j)
        prox = net.social_proximity(i, j)
        norm = pop.culture[i, 0] - 0.5
        scarcity = pop.scarcity()[i]
        own = np.tanh((w[i] - w_ref) / w_ref)
        gap = np.tanh((w[j] - w[i]) / w_ref)

        grp = pop.belief_group[i, pop.ethnicity[j]]
        dy = np.array([self.dyad[int(a)].get(int(b), 0.0) for a, b in zip(i, j)])

        return {
            "base": np.full(len(i), cfg.beta_0),
            "ethnic": beta_eth_i * (same_eth - self.p_same_ethnicity),
            "cultural": cfg.beta_cultural * (cult_sim - 0.5),
            "social": cfg.beta_social * prox,
            "norm": cfg.beta_norm * norm,
            "own_wealth": cfg.beta_own_wealth * own,
            "wealth_gap": cfg.beta_wealth_gap * gap,
            "scarcity": cfg.beta_scarcity * scarcity,
            "group_belief": cfg.lambda_group * grp,
            "dyad_belief": cfg.lambda_dyad * dy,
        }

    def p_cooperate(
        self, i: np.ndarray, j: np.ndarray, net: SocialNetwork, rng: np.random.Generator
    ) -> np.ndarray:
        comp = self.components(i, j, net)
        z = sum(comp.values())
        if self.cfg.decision_noise > 0:
            z = z + rng.normal(0.0, self.cfg.decision_noise, size=len(i))
        return sigmoid(z)

    # ------------------------------------------------------------------
    def update(
        self, i: np.ndarray, j: np.ndarray, signal_from_j: np.ndarray
    ) -> None:
        """Rescorla-Wagner update of i's beliefs given j's behaviour.

        `signal_from_j` is in [-1, 1]: +1 fully cooperative, -1 fully defecting.
        """
        cfg, pop = self.cfg, self.pop
        gj = pop.ethnicity[j]
        cur = pop.belief_group[i, gj]
        pop.belief_group[i, gj] = cur + cfg.alpha_group * (signal_from_j - cur)

        a_d = cfg.alpha_dyad
        for a, b, s in zip(i.tolist(), j.tolist(), signal_from_j.tolist()):
            d = self.dyad[a]
            d[b] = d.get(b, 0.0) + a_d * (s - d.get(b, 0.0))

    def decay_dyads(self, rate: float = 0.01) -> None:
        """Memory of specific partners fades between encounters."""
        if rate <= 0:
            return
        keep = 1.0 - rate
        for d in self.dyad.values():
            for k in list(d.keys()):
                v = d[k] * keep
                if abs(v) < 1e-3:
                    del d[k]
                else:
                    d[k] = v
