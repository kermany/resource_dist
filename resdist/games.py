"""The two interaction games.

`donation`  - simultaneous prisoner's-dilemma-style giving. Symmetric.
`trust`     - sequential investment game. Asymmetric: an investor stakes part
              of their holdings, the stake is multiplied, and the trustee
              chooses how much of the pot to return. Role assignment is biased
              toward the wealthier agent taking the investor role, which makes
              this the patron-client channel.

Both return per-agent wealth deltas, the [-1, 1] behavioural signal each side
observed from the other, and a [0, 1] cooperation intensity used for metrics.
"""

from __future__ import annotations

import numpy as np

from .config import Config


class GameOutcome:
    __slots__ = ("delta", "sig_i", "sig_j", "coop_i", "coop_j")

    def __init__(self, delta, sig_i, sig_j, coop_i, coop_j):
        self.delta = delta      # (N,) wealth change
        self.sig_i = sig_i      # signal that i observed from j, in [-1, 1]
        self.sig_j = sig_j      # signal that j observed from i
        self.coop_i = coop_i    # cooperation intensity of i toward j, [0, 1]
        self.coop_j = coop_j


def play_donation(
    cfg: Config,
    wealth: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    rng: np.random.Generator,
) -> GameOutcome:
    C_i = (rng.random(len(i)) < p_i).astype(float)
    C_j = (rng.random(len(j)) < p_j).astype(float)

    if cfg.proportional_cost:
        cost_i = cfg.donation_cost * wealth[i]
        cost_j = cfg.donation_cost * wealth[j]
    else:
        cost_i = np.full(len(i), cfg.donation_cost)
        cost_j = np.full(len(j), cfg.donation_cost)

    # you cannot give what you do not have
    cost_i = np.minimum(cost_i, np.maximum(wealth[i], 0.0))
    cost_j = np.minimum(cost_j, np.maximum(wealth[j], 0.0))
    paid_i = cost_i * C_i
    paid_j = cost_j * C_j

    delta = np.zeros_like(wealth)
    np.add.at(delta, i, -paid_i + cfg.donation_mult * paid_j)
    np.add.at(delta, j, -paid_j + cfg.donation_mult * paid_i)

    return GameOutcome(delta, 2.0 * C_j - 1.0, 2.0 * C_i - 1.0, C_i, C_j)


def play_trust(
    cfg: Config,
    wealth: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    rng: np.random.Generator,
) -> GameOutcome:
    """i and j are assigned investor / trustee roles by wealth (mostly)."""
    n = len(i)
    rich_first = wealth[i] >= wealth[j]
    by_wealth = rng.random(n) < cfg.p_rich_is_investor
    coin = rng.random(n) < 0.5
    i_is_investor = np.where(by_wealth, rich_first, coin)

    inv = np.where(i_is_investor, i, j)
    tru = np.where(i_is_investor, j, i)
    p_inv = np.where(i_is_investor, p_i, p_j)   # willingness to trust
    p_tru = np.where(i_is_investor, p_j, p_i)   # trustworthiness

    stake_base = np.minimum(np.maximum(wealth[inv], 0.0), cfg.trust_stake_cap)
    endow = cfg.trust_stake_frac * stake_base
    sent = p_inv * endow
    pot = cfg.trust_mult * sent
    rho = cfg.trust_return_max * p_tru
    returned = pot * rho

    delta = np.zeros_like(wealth)
    np.add.at(delta, inv, -sent + returned)
    np.add.at(delta, tru, pot - returned)

    # cooperation intensity: share of endowment staked / share of pot returned
    with np.errstate(invalid="ignore", divide="ignore"):
        send_frac = np.where(endow > 1e-12, sent / np.maximum(endow, 1e-12), p_inv)
    send_frac = np.clip(send_frac, 0.0, 1.0)
    ret_frac = np.clip(rho, 0.0, 1.0)

    sig_from_inv = 2.0 * send_frac - 1.0   # what the trustee observed
    sig_from_tru = 2.0 * ret_frac - 1.0    # what the investor observed

    # map back to the (i, j) orientation
    coop_i = np.where(i_is_investor, send_frac, ret_frac)
    coop_j = np.where(i_is_investor, ret_frac, send_frac)
    sig_i = np.where(i_is_investor, sig_from_tru, sig_from_inv)
    sig_j = np.where(i_is_investor, sig_from_inv, sig_from_tru)

    return GameOutcome(delta, sig_i, sig_j, coop_i, coop_j)
