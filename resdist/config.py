"""Parameter schema for the resource-distribution / cooperation model.

Every knob the model exposes lives here. A Config is a plain dataclass so it
serialises to JSON cleanly and can be swept programmatically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Population structure
    # ------------------------------------------------------------------
    n_agents: int = 600
    n_groups: int = 3
    #: Relative sizes of the ethnic groups; normalised internally.
    group_shares: Sequence[float] = (0.5, 0.3, 0.2)
    #: Mean initial resources per group. Structural inequality at t=0.
    group_wealth_means: Sequence[float] = (12.0, 10.0, 8.0)
    #: Lognormal dispersion of initial resources within a group.
    wealth_sigma: float = 0.45
    #: Per-group multiplier on income-generating capacity. This is structural
    #: advantage in *access to returns* (land, credit, licences, schooling)
    #: as distinct from the initial stock of resources. Without it a single
    #: global attractor erases all group differences within ~100 rounds.
    group_productivity: Sequence[float] = (1.2, 1.0, 0.75)
    #: Lognormal spread of individual productivity within a group.
    productivity_sigma: float = 0.35

    # ------------------------------------------------------------------
    # Culture
    # ------------------------------------------------------------------
    #: Dimensionality of the cultural trait vector. Trait 0 is the
    #: "cooperative norm" trait and feeds the decision rule directly.
    n_culture_dims: int = 4
    #: Distance between group cultural centroids (0 = identical cultures).
    culture_group_separation: float = 0.35
    #: Within-group spread of cultural traits.
    culture_sigma: float = 0.12
    #: Rate at which agents shift culture toward partners after a mutually
    #: cooperative interaction (0 disables cultural transmission).
    culture_drift: float = 0.01

    # ------------------------------------------------------------------
    # Social network
    # ------------------------------------------------------------------
    mean_degree: float = 8.0
    #: Strength of ethnic homophily in initial tie formation.
    homophily_ethnic: float = 1.5
    #: Strength of cultural homophily in initial tie formation.
    homophily_cultural: float = 1.2
    #: Strength of wealth homophily (assortative mixing by resources).
    homophily_wealth: float = 0.8
    #: Probability that a given agent revises one tie per round.
    rewire_rate: float = 0.02
    #: Probability an interaction partner is drawn from the network rather
    #: than from the population at large.
    p_local: float = 0.75

    # ------------------------------------------------------------------
    # Decision rule: P(cooperate) = sigmoid(z)
    # ------------------------------------------------------------------
    beta_0: float = -0.6           # baseline propensity
    beta_ethnic: float = 1.2       # same-ethnicity bonus
    #: Optional per-group override of `beta_ethnic`, e.g. to give one group an
    #: elevated (or suppressed) taste for co-ethnics relative to the others.
    #: None means every group uses the scalar `beta_ethnic` above.
    group_beta_ethnic: Sequence[float] | None = None
    beta_cultural: float = 1.0     # cultural similarity (centred)
    beta_social: float = 0.6       # social proximity (tie + shared neighbours)
    beta_norm: float = 1.5         # own cooperative-norm trait (centred)
    beta_own_wealth: float = 0.5   # affluence -> more willing to give
    beta_wealth_gap: float = -0.6  # partner richer than me -> less willing
    beta_scarcity: float = -2.5    # near-destitution -> desperation defection
    #: Scarcity is measured against a buffer of this many rounds of subsistence.
    scarcity_buffer_rounds: float = 5.0
    #: Weight on the learned group-level reputation term.
    lambda_group: float = 1.0
    #: Weight on the learned dyad-level (direct reciprocity) term.
    lambda_dyad: float = 1.2
    #: Learning rates for the two belief systems.
    alpha_group: float = 0.02
    alpha_dyad: float = 0.30
    #: Idiosyncratic noise on the latent propensity each interaction.
    decision_noise: float = 0.25

    # ------------------------------------------------------------------
    # Games
    # ------------------------------------------------------------------
    #: Share of interactions that are asymmetric trust games; the rest are
    #: simultaneous donation / prisoner's dilemma games.
    trust_game_frac: float = 0.35
    # -- donation game --
    donation_cost: float = 0.6     # absolute cost paid by a cooperator
    donation_mult: float = 2.4     # benefit = mult * cost
    #: If True the cost is a fraction of the donor's wealth instead of absolute.
    proportional_cost: bool = False
    # -- trust game --
    trust_stake_frac: float = 0.25  # max share of wealth an investor can send
    #: Absolute ceiling on the wealth an investor's stake is computed from, so
    #: that per-interaction surplus stays bounded and the economy does not
    #: compound explosively.
    trust_stake_cap: float = 10.0
    trust_mult: float = 2.5         # multiplier applied to the sent amount
    trust_return_max: float = 1.0   # max share of the pot a trustee can return
    #: Probability the wealthier agent takes the investor role (patron-client).
    p_rich_is_investor: float = 0.8

    # ------------------------------------------------------------------
    # Resource dynamics
    # ------------------------------------------------------------------
    # Solow-style: flat income plus a diminishing return on holdings, against
    # a fixed subsistence draw and proportional depreciation. Calibrated so a
    # non-cooperating agent settles near w ~ 10 and a fully cooperating one
    # near w ~ 20, which keeps the scarcity term live instead of vestigial.
    base_income: float = 1.2        # flat income each round
    capital_return: float = 0.30   # return on held resources
    capital_exponent: float = 0.6   # <1 => diminishing returns to wealth
    subsistence: float = 1.6        # per-round consumption requirement
    depreciation: float = 0.10      # per-round decay of holdings
    income_shock_sigma: float = 0.15  # lognormal multiplicative income shock
    #: Idiosyncratic crises (illness, crop failure, expropriation). These are
    #: what push agents into the region where the scarcity term bites; the
    #: wealthy absorb them, the thinly-buffered do not.
    crisis_prob: float = 0.02
    crisis_severity: float = 0.55     # share of holdings destroyed by a crisis
    #: Flat tax on income, redistributed as a uniform per-capita transfer.
    tax_rate: float = 0.0

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------
    n_rounds: int = 400
    seed: int = 0
    #: Record metrics every k rounds.
    record_every: int = 5
    #: Horizon (in rounds) for the short-run mobility measure. Persistence
    #: against t=0 washes out in a mean-reverting economy; this measures
    #: whether the *current* ordering is sticky.
    mobility_lag: int = 100
    label: str = "baseline"

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)

    def variant(self, **kwargs) -> "Config":
        """Return a copy with the given fields overridden."""
        return replace(self, **kwargs)

    def validate(self) -> None:
        assert self.n_groups == len(self.group_shares), "group_shares length must equal n_groups"
        assert self.n_groups == len(self.group_wealth_means), "group_wealth_means length must equal n_groups"
        assert self.n_groups == len(self.group_productivity), "group_productivity length must equal n_groups"
        if self.group_beta_ethnic is not None:
            assert self.n_groups == len(self.group_beta_ethnic), "group_beta_ethnic length must equal n_groups"
        assert self.n_agents > 1
        assert 0.0 <= self.trust_game_frac <= 1.0
        assert 0.0 <= self.p_local <= 1.0
        assert 0.0 <= self.tax_rate < 1.0
        assert self.n_culture_dims >= 1
