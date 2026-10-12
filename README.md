# resdist — resource distribution under group-structured cooperation

An agent-based model of how resources come to be distributed in a population whose
members play cooperative games with one another, where the probability of
cooperating depends on ethnic, cultural and social distance **and** on the players'
current economic position.

The point of the model is that these channels interact. Identity shapes who you
meet and how you treat them; poverty shapes whether you can afford to cooperate at
all; and how you were treated yesterday shapes what you believe about a whole group
tomorrow. Each of those can be switched off independently, so their contributions
can be separated rather than asserted.

```
resdist/
  config.py       every parameter, as one dataclass
  population.py   agent state (ethnicity, culture, wealth, productivity, beliefs)
  network.py      homophilous tie formation, endogenous rewiring, round matching
  decision.py     the hybrid cooperation rule
  games.py        donation game + asymmetric trust game
  metrics.py      Gini, Theil decomposition, assortativity, mobility
  model.py        the simulation loop
tests/test_core.py       40 checks: closed-form metrics, payoff accounting, invariants
run_experiments.py       9 scenarios x 5 seeds + two 2-D parameter sweeps
build_dashboard.py       assembles the interactive HTML viewer
```

Quick start:

```bash
python3 tests/test_core.py        # ~25s, all checks
python3 run_experiments.py        # ~3.5 min on 2 cores -> outputs/*.json
python3 build_dashboard.py        # -> outputs/resource_distribution_dashboard.html
```

```python
from resdist import Config, Simulation
sim = Simulation(Config(n_agents=600, n_rounds=400, beta_ethnic=1.2, seed=0))
hist = sim.run()                  # list of per-snapshot metric dicts
sim.cooperation_matrix()          # 3x3 mean cooperation, row group -> column group
```

---

## 1. Agents

Each agent `i` carries:

| State | Meaning |
|---|---|
| `ethnicity ∈ {0..G-1}` | immutable group label; group sizes 50 / 30 / 20 % |
| `culture ∈ [0,1]^K` | trait vector drawn around a group centroid; dim 0 is the *cooperative norm* trait |
| `wealth ≥ 0` | current resource holdings |
| `productivity` | multiplier on income — structural access to returns (land, credit, schooling), distinct from the initial stock |
| `belief_group ∈ [-1,1]^G` | learned valence toward each ethnic group — the stereotype channel |
| `dyad[j] ∈ [-1,1]` | learned valence toward a specific partner — direct reciprocity |

## 2. The cooperation rule

`P(cooperate) = σ(z)`, with the latent propensity built from a structural baseline
plus two learned terms:

```
z_ij =  β₀
      + β_eth  · (same_ethnicity − p_same)      ← centred, see note
      + β_cult · (cultural_similarity − ½)
      + β_soc  · social_proximity                ← tie indicator + neighbourhood Jaccard
      + β_norm · (own cooperative-norm trait − ½)
      + β_own  · tanh((wᵢ − w̄)/w̄)              ← affluence makes giving cheap
      + β_gap  · tanh((wⱼ − wᵢ)/w̄)              ← resentment of a richer partner
      + β_scar · scarcity_i                      ← desperation, in [0,1]
      + λ_group · belief_group_i[eth_j]
      + λ_dyad  · dyad_i[j]
      + ε,  ε ~ N(0, σ²)
```

**Why the ethnic term is centred.** `same_ethnicity` is centred on the probability
that two randomly drawn agents share an ethnicity. Without this, raising
`β_eth` also raises the *average* propensity to cooperate, so a "no ethnic bias"
counterfactual would silently double as a "less cooperation overall"
counterfactual and the two effects could not be told apart. Centred, `β_eth`
controls the in-group/out-group *gap* and leaves the level alone. This mattered:
uncentred, switching bias off appeared to *raise* inequality, purely because it
had gutted total cooperation.

Learning is Rescorla–Wagner on the observed behaviour signal `s ∈ [-1,1]`:

```
belief ← belief + α · (s − belief)
```

with a slow rate for groups (`α = 0.02`) and a fast one for individuals
(`α = 0.30`), so stereotypes accumulate gradually and are stickier than
first-hand impressions. Dyadic memory decays 0.5 % per round.

## 3. The games

A share `trust_game_frac` of interactions are trust games; the rest are donation games.

**Donation game** (symmetric). A cooperator pays cost `c`; the partner receives
`m·c` with `m > 1`. An agent cannot pay a cost it cannot afford — poverty is a
hard constraint on cooperation, not merely a disincentive.

**Trust game** (asymmetric, the patron–client channel). The wealthier agent takes
the investor role with probability `p_rich_is_investor`. The investor stakes
`P(coop) × trust_stake_frac × min(w, cap)`, the stake is multiplied by `m`, and
the trustee returns a share equal to its own `P(coop)`. The stake is capped in
absolute terms so per-interaction surplus stays bounded.

## 4. The economy

Solow-style, so that the resource level is a stationary attractor rather than an
exponential:

```
income  = productivity × (base + r·w^η) × lognormal shock      η < 1
w      ← w + income − (crisis loss) + (game payoffs) − subsistence,  then ×(1−δ)
```

Crises destroy 55 % of holdings with probability `crisis_prob` per round. They are
what push agents into the region where the scarcity term bites; the wealthy absorb
them and the thinly-buffered do not. **Calibration is load-bearing here.** The
first calibration let the economy compound at ~4 % per round, so within 200 rounds
everyone was rich, the scarcity term never activated, and the poverty channel was
dead code. The second over-corrected: a single global attractor erased all group
differences within ~100 rounds. Heterogeneous productivity plus crisis shocks is
what makes both inequality and the scarcity channel live.

At zero wealth, income falls short of subsistence for most agents, so **escaping
destitution requires being cooperated with** — which is exactly what the scarcity
term makes unlikely.

## 5. Measures

- **Gini** of holdings, and **Theil T** additively decomposed into between-group
  and within-group components (the decomposition identity is unit-tested).
- **Cooperation rate**, split into in-group and out-group, and the 3×3 matrix of
  group-toward-group cooperation.
- **Newman assortativity** of the tie network by ethnicity; cross-group tie share.
- **Rank persistence** against t=0 and against 100 rounds earlier (short-run
  stickiness — the long-run measure washes out in a mean-reverting economy).

## 6. What the baseline produces

500 agents, 350 rounds, 5 seeds.

| | Baseline |
|---|---|
| Gini | 0.58 |
| Between-group share of inequality | 22.3 % |
| In-group cooperation bias | 0.52 |
| Network ethnic assortativity | 0.62 |
| Share holding nothing | 8.2 % |

Group C is 20 % of the population and ends holding **4.9 %** of all resources, with
30 % of its members at zero. Its cooperation collapses toward *everyone* — its
in-group cooperation falls to 0.41 against Group A's 0.84. That collapse is not
identity, it is poverty: under equal access to returns, Group C's in-group
cooperation is 0.74, essentially Group A's 0.77.

### Counterfactuals (Δ from baseline)

| Scenario | Gini | Between-share | Coop | In-group bias | Assortativity |
|---|---|---|---|---|---|
| Baseline | 0.578 | 0.223 | 0.518 | 0.522 | 0.620 |
| No taste-based bias (`β_eth=0`) | 0.573 | 0.210 | 0.438 | **0.175** | 0.551 |
| No ethnic homophily in ties | 0.600 | 0.199 | 0.420 | 0.497 | **0.144** |
| No stereotype learning (`λ_group=0`) | 0.550 | 0.193 | 0.519 | 0.399 | 0.602 |
| No desperation (`β_scar=0`) | 0.559 | 0.164 | 0.644 | 0.595 | 0.628 |
| Equal access to returns | 0.526 | **0.015** | 0.540 | 0.515 | 0.593 |
| Equal access + no ethnic bias | 0.538 | 0.010 | 0.432 | 0.036 | 0.039 |
| Redistribution (35 % tax) | **0.304** | 0.138 | 0.652 | 0.598 | 0.626 |
| Forced contact (mostly strangers) | 0.582 | 0.202 | 0.377 | 0.456 | 0.568 |

Four results worth stating plainly:

1. **Identity bias alone produces almost no between-group inequality.** With equal
   access to returns, the between-group share collapses from 22 % to 1.5 % *even
   though in-group bias remains at 0.52*. Symmetric favouritism is close to a wash
   across groups; what opens a durable gap is differential access to returns. The
   residual minority penalty is real but an order of magnitude smaller — Group C
   takes 18.8 % of resources on 20 % of the population.
2. **Preference and segregation are near-orthogonal channels.** Removing the taste
   for co-ethnics cuts the cooperation bias (0.52 → 0.18) but barely dents network
   segregation (0.62 → 0.55). Removing ethnic homophily in tie formation collapses
   segregation (0.62 → 0.14) but barely dents the bias (0.52 → 0.50). Neither
   intervention is a substitute for the other.
3. **Stereotype learning amplifies rather than creates.** Switching off group-level
   generalisation cuts the bias by about a quarter (0.52 → 0.40) — it takes an
   existing preference and compounds it into a self-fulfilling one, but it does not
   manufacture bias from nothing (see the sweep: at `β_eth = 0` the bias stays near
   0.11–0.18 no matter how strong the learning).
4. **Redistribution raises cooperation.** A 35 % tax cuts the Gini from 0.58 to 0.30,
   eliminates destitution, *and* raises the cooperation rate from 0.52 to 0.65, by
   removing the desperation-driven defection that poverty was generating. It also
   raises measured in-group bias to 0.60 — with indiscriminate poverty-defection
   gone, what remains visible is pure identity preference.

Contact policy is the cautionary one: forcing interaction with strangers lowers
cooperation (0.52 → 0.38) while leaving bias largely intact, because the structural
preference is untouched by exposure alone.

### Sweeps

- `β_eth × λ_group` on in-group bias: bias runs 0.11 → 0.82. The two multiply —
  learning has almost no effect at zero preference, and steep effect at high
  preference.
- `β_scar × crisis_prob` on in-group bias: bias *falls* from 0.58 to 0.28 as
  desperation and shock exposure rise together, because desperate agents defect on
  co-ethnics too. Economic stress dissolves the in-group premium rather than
  sharpening it — the poor stop being able to afford solidarity.

## 7. Limitations

- Groups differ only in size, initial wealth and productivity. There is no
  asymmetric discrimination (every group favours its own equally), no institutional
  layer, no coalition formation, and no intergenerational transmission.
- Cooperation "intensity" mixes a binary donation decision with a continuous trust
  stake; the aggregate rate is a mean of the two.
- Parameters are calibrated for qualitative plausibility, not fitted to data. The
  numbers above are properties of *this* parameterisation — the counterfactual
  *directions* are the robust part, the magnitudes are not.
- `culture_drift` is on by default but weak; cultural convergence is not a
  substantively explored channel yet.

Natural next steps: asymmetric discrimination (a group everyone defects on),
inheritance across generations, group-conditional institutions, and fitting the
decision weights to experimental-game data.
