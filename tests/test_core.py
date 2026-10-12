"""Sanity and correctness tests for the engine."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from resdist import Config, Simulation  # noqa: E402
from resdist import metrics  # noqa: E402
from resdist.decision import sigmoid  # noqa: E402


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def tail(hist, key, k=10):
    """Average a metric over the last k recorded snapshots.

    Single-snapshot reads are noisy: each record averages only `record_every`
    rounds, which for a rate measured on ~400 interactions has a standard
    error around 0.013.
    """
    vals = [r[key] for r in hist[-k:] if key in r]
    return float(np.nanmean(vals)) if vals else float("nan")


# ----------------------------------------------------------------------
print("\n[metrics] closed-form checks")

check("gini of equal vector is 0", approx(metrics.gini(np.ones(100)), 0.0, 1e-9))
# perfectly unequal: one agent holds everything -> (n-1)/n
x = np.zeros(100); x[0] = 1.0
check("gini of maximal inequality ~ (n-1)/n", approx(metrics.gini(x), 0.99, 1e-9),
      f"got {metrics.gini(x)}")
# known value: uniform 1..n has gini (n-1)/(3n)
n = 1000
u = np.arange(1, n + 1, dtype=float)
check("gini of 1..n matches (n-1)/(3n)", approx(metrics.gini(u), (n - 1) / (3 * n), 1e-3),
      f"got {metrics.gini(u)} vs {(n-1)/(3*n)}")

check("theil of equal vector is 0", approx(metrics.theil_t(np.full(50, 7.0)), 0.0, 1e-12))

# Theil decomposition: two groups, no within-group variance -> all between
w = np.array([2.0] * 50 + [8.0] * 50)
g = np.array([0] * 50 + [1] * 50)
d = metrics.theil_decomposition(w, g)
check("theil decomposition sums to total", approx(d["between"] + d["within"], d["total"], 1e-12))
check("no within-group variance -> between_share == 1", approx(d["between_share"], 1.0, 1e-9),
      f"got {d['between_share']}")
check("decomposed total equals direct theil", approx(d["total"], metrics.theil_t(w), 1e-10),
      f"{d['total']} vs {metrics.theil_t(w)}")

# same group everywhere -> all within
d2 = metrics.theil_decomposition(w, np.zeros(100, dtype=int))
check("single group -> between_share == 0", approx(d2["between_share"], 0.0, 1e-9))

# assortativity: fully segregated graph -> 1.0
edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
attr = np.array([0, 0, 0, 1, 1, 1])
check("assortativity of segregated graph == 1", approx(metrics.categorical_assortativity(edges, attr), 1.0, 1e-9),
      f"got {metrics.categorical_assortativity(edges, attr)}")
# fully cross-cutting bipartite graph -> negative
edges2 = np.array([[0, 3], [1, 4], [2, 5]])
check("assortativity of cross-cutting graph < 0", metrics.categorical_assortativity(edges2, attr) < 0)
check("cross_group_tie_share correct", approx(metrics.cross_group_tie_share(edges2, attr), 1.0))

check("rank_correlation self == 1", approx(metrics.rank_correlation(u, u), 1.0, 1e-9))
check("rank_correlation reversed == -1", approx(metrics.rank_correlation(u, -u), -1.0, 1e-9))

# tail concentration: 100 agents, one holds 91, the rest hold 1 each -> top 1%
# (the single agent) holds 91 / (91 + 99) of the total
tx = np.concatenate([[91.0], np.ones(99)])
check("top_share picks out the single richest agent",
      approx(metrics.top_share(tx, 0.01), 91.0 / 190.0, 1e-9), f"got {metrics.top_share(tx, 0.01)}")
check("bottom_share of an equal vector is proportionate",
      approx(metrics.bottom_share(np.ones(100), 0.10), 0.10, 1e-9))
check("top half + bottom half shares partition the whole vector",
      approx(metrics.top_share(tx, 0.5) + metrics.bottom_share(tx, 0.5), 1.0, 1e-9),
      f"got {metrics.top_share(tx, 0.5) + metrics.bottom_share(tx, 0.5)}")
check("tail_mean(top) of 1..100 is the mean of 91..100",
      approx(metrics.tail_mean(np.arange(1, 101, dtype=float), 0.10, "top"), np.arange(91, 101).mean(), 1e-9))
check("tail_mean(bottom) of 1..100 is the mean of 1..10",
      approx(metrics.tail_mean(np.arange(1, 101, dtype=float), 0.10, "bottom"), np.arange(1, 11).mean(), 1e-9))

# representation: group 1 holds the entire top decile -> 1/0.5 = 2x its share
rw = np.concatenate([np.full(50, 100.0), np.full(50, 1.0)])
rg = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
rep = metrics.tail_group_representation(rw, rg, 0.10, "top")
check("group holding the whole top decile is 2x over-represented (half the population)",
      approx(rep[1], 2.0, 1e-9), f"got {rep}")
check("group absent from the top decile has representation 0",
      approx(rep[0], 0.0, 1e-9), f"got {rep}")

# ----------------------------------------------------------------------
print("\n[decision] rule behaves monotonically")
check("sigmoid(0) == 0.5", approx(float(sigmoid(np.array([0.0]))[0]), 0.5))

cfg = Config(n_agents=200, n_rounds=1, seed=1)
sim = Simulation(cfg)
i = np.arange(0, 50)
j = np.arange(50, 100)
comp = sim.dec.components(i, j, sim.net)
same = (sim.pop.ethnicity[i] == sim.pop.ethnicity[j])
check("ethnic term favours co-ethnics over out-group",
      np.all(comp["ethnic"][same] > 0) and np.all(comp["ethnic"][~same] < 0))
# centring means the term must not move the population-average propensity
p_same = sim.dec.p_same_ethnicity
check("ethnic term is mean-zero under random matching",
      approx(p_same * cfg.beta_ethnic * (1 - p_same)
             + (1 - p_same) * cfg.beta_ethnic * (0 - p_same), 0.0, 1e-12))
check("scarcity term is non-positive", np.all(comp["scarcity"] <= 0))

# group-specific ethnic bias: one group's co-ethnic preference can be
# elevated relative to the others without touching the centring constant
asym = Config(n_agents=300, n_rounds=1, seed=1, beta_ethnic=1.2,
              group_beta_ethnic=(1.2, 1.2, 2.4))
asim = Simulation(asym)
check("per-group beta_ethnic overrides the scalar default",
      np.allclose(asim.dec.beta_ethnic_by_group, [1.2, 1.2, 2.4]))
check("centring constant is unaffected by the per-group override",
      approx(asim.dec.p_same_ethnicity, sim.dec.p_same_ethnicity, 1e-9))
i2 = np.where(asim.pop.ethnicity == 2)[0][:20]
j2 = np.where(asim.pop.ethnicity == 2)[0][20:40]
comp2 = asim.dec.components(i2, j2, asim.net)
expected_g2 = 2.4 * (1 - asim.dec.p_same_ethnicity)
check("elevated group's in-group ethnic term matches its own beta_ethnic",
      np.allclose(comp2["ethnic"], expected_g2), f"got {comp2['ethnic'][:3]} vs {expected_g2}")

# turning off every driver leaves an unbiased coin flip
flat = Config(n_agents=200, n_rounds=1, seed=2, beta_0=0.0, beta_ethnic=0.0,
              beta_cultural=0.0, beta_social=0.0, beta_norm=0.0,
              beta_own_wealth=0.0, beta_wealth_gap=0.0, beta_scarcity=0.0,
              lambda_group=0.0, lambda_dyad=0.0, decision_noise=0.0)
s2 = Simulation(flat)
p = s2.dec.p_cooperate(np.arange(100), np.arange(100, 200), s2.net, s2.rng)
check("all drivers off -> P(coop) == 0.5 everywhere", np.allclose(p, 0.5), f"range {p.min()}-{p.max()}")

# ----------------------------------------------------------------------
print("\n[games] payoff accounting")
from resdist import games  # noqa: E402

gcfg = Config(donation_cost=1.0, donation_mult=2.0, proportional_cost=False)
wealth = np.full(4, 10.0)
i = np.array([0, 2]); j = np.array([1, 3])
rng = np.random.default_rng(0)
# force full cooperation
out = games.play_donation(gcfg, wealth, i, j, np.ones(2), np.ones(2), rng)
check("mutual cooperation creates surplus (mult-1)*cost each",
      np.allclose(out.delta[[0, 1, 2, 3]], 1.0), f"got {out.delta}")
out = games.play_donation(gcfg, wealth, i, j, np.zeros(2), np.zeros(2), rng)
check("mutual defection changes nothing", np.allclose(out.delta, 0.0))
out = games.play_donation(gcfg, wealth, i, j, np.ones(2), np.zeros(2), rng)
check("sucker pays cost, defector reaps benefit",
      np.allclose(out.delta[[0, 2]], -1.0) and np.allclose(out.delta[[1, 3]], 2.0),
      f"got {out.delta}")
check("defector's signal to the cooperator is -1", np.allclose(out.sig_i, -1.0))

# a broke agent cannot pay
poor = np.array([0.0, 10.0])
out = games.play_donation(gcfg, poor, np.array([0]), np.array([1]), np.ones(1), np.zeros(1), rng)
check("agent with zero wealth cannot pay a cost", out.delta[0] == 0.0, f"got {out.delta}")

# trust game conservation: total wealth change == (mult-1)*sent
tcfg = Config(trust_mult=3.0, trust_stake_frac=0.5, trust_return_max=1.0, p_rich_is_investor=1.0)
wealth = np.array([10.0, 4.0])
out = games.play_trust(tcfg, wealth, np.array([0]), np.array([1]), np.array([1.0]), np.array([0.5]), rng)
sent = 0.5 * 10.0 * 1.0
check("trust game surplus == (m-1)*sent", approx(out.delta.sum(), (3.0 - 1.0) * sent, 1e-9),
      f"got {out.delta.sum()} vs {(3.0-1.0)*sent}")
check("richer agent took the investor role", out.delta[0] == -sent + 3.0 * sent * 0.5)

# ----------------------------------------------------------------------
print("\n[dynamics] structural invariants over a full run")

base = Config(n_agents=300, n_rounds=120, seed=7, label="test")
sim = Simulation(base)
hist = sim.run()
check("history recorded", len(hist) > 0)
check("wealth never negative", np.all(sim.pop.wealth >= 0.0))
check("beliefs stay in [-1, 1]", np.all(np.abs(sim.pop.belief_group) <= 1.0 + 1e-9))
check("culture stays in [0, 1]", np.all((sim.pop.culture >= 0) & (sim.pop.culture <= 1)))
check("gini is a valid coefficient", 0.0 <= hist[-1]["gini"] <= 1.0)
check("all agents interact", sim.pop.n_interactions.min() > 0)
check("theil identity holds in the time series",
      all(approx(r["theil_between"] + r["theil_within"], r["theil_total"], 1e-9) for r in hist))

# ---- degenerate case: no ethnic bias, identical groups, no learning ----
neutral = Config(
    n_agents=400, n_rounds=200, seed=3, label="neutral",
    group_wealth_means=(10.0, 10.0, 10.0),
    group_productivity=(1.0, 1.0, 1.0),
    culture_group_separation=0.0,
    homophily_ethnic=0.0, homophily_cultural=0.0, homophily_wealth=0.0,
    beta_ethnic=0.0, lambda_group=0.0,
)
ns = Simulation(neutral)
nh = ns.run()
check("symmetric world -> assortativity near zero", abs(nh[-1]["assortativity"]) < 0.10,
      f"got {nh[-1]['assortativity']}")
check("symmetric world -> in-group bias near zero", abs(tail(nh, "ingroup_bias")) < 0.04,
      f"got {tail(nh, 'ingroup_bias')}")
check("symmetric world -> between-group share of inequality is small",
      nh[-1]["between_share"] < 0.05, f"got {nh[-1]['between_share']}")

# ---- ethnic bias on -> segregation and bias should appear ----------
biased = neutral.variant(label="biased", beta_ethnic=2.0, homophily_ethnic=3.0,
                         lambda_group=1.5, seed=3)
bs = Simulation(biased)
bh = bs.run()
check("ethnic homophily raises network assortativity",
      bh[-1]["assortativity"] > nh[-1]["assortativity"] + 0.2,
      f"{bh[-1]['assortativity']} vs {nh[-1]['assortativity']}")
check("ethnic bias raises in-group cooperation bias",
      tail(bh, "ingroup_bias") > tail(nh, "ingroup_bias") + 0.05,
      f"{tail(bh, 'ingroup_bias')} vs {tail(nh, 'ingroup_bias')}")

# ---- scarcity sensitivity: stronger desperation -> less cooperation ----
# Use a harsher economy so a meaningful share of agents sit in the scarcity
# region; with an affluent population the term is correctly inert.
lean = dict(n_agents=300, n_rounds=250, seed=11, base_income=0.9,
            capital_return=0.18, crisis_prob=0.05)
soft = Config(label="soft", beta_scarcity=0.0, **lean)
hard = Config(label="hard", beta_scarcity=-6.0, **lean)
s_soft, s_hard = Simulation(soft), Simulation(hard)
sh, hh = s_soft.run(), s_hard.run()
check("test economy actually puts agents under scarcity",
      s_soft.pop.scarcity().mean() > 0.05, f"mean scarcity {s_soft.pop.scarcity().mean()}")
check("stronger scarcity penalty lowers cooperation",
      tail(hh, "coop_rate") < tail(sh, "coop_rate") - 0.01,
      f"{tail(hh, 'coop_rate')} vs {tail(sh, 'coop_rate')}")
check("scarcity-driven defection deepens destitution",
      hh[-1]["destitute_frac"] >= sh[-1]["destitute_frac"],
      f"{hh[-1]['destitute_frac']} vs {sh[-1]['destitute_frac']}")

# ---- crises are what generate the poor tail ------------------------
calm = Config(n_agents=300, n_rounds=250, seed=13, crisis_prob=0.0, label="calm")
rough = calm.variant(crisis_prob=0.06, label="rough")
ch = Simulation(calm).run()
rh = Simulation(rough).run()
check("crisis shocks raise inequality", rh[-1]["gini"] > ch[-1]["gini"],
      f"{rh[-1]['gini']} vs {ch[-1]['gini']}")

# ---- structural productivity advantage drives the between-group gap ----
flatp = Config(n_agents=400, n_rounds=250, seed=17, label="flatp",
               group_productivity=(1.0, 1.0, 1.0), productivity_sigma=0.35)
skewp = flatp.variant(group_productivity=(1.3, 1.0, 0.7), label="skewp")
fh = Simulation(flatp).run()
kh = Simulation(skewp).run()
check("unequal group productivity raises the between-group share",
      kh[-1]["between_share"] > fh[-1]["between_share"] + 0.05,
      f"{kh[-1]['between_share']} vs {fh[-1]['between_share']}")
check("equal productivity leaves little between-group inequality",
      fh[-1]["between_share"] < 0.06, f"got {fh[-1]['between_share']}")

# ---- redistribution reduces inequality ----------------------------
notax = Config(n_agents=300, n_rounds=200, seed=5, tax_rate=0.0, label="notax")
tax = notax.variant(tax_rate=0.4, label="tax")
a = Simulation(notax).run()
b = Simulation(tax).run()
check("redistribution lowers the Gini", b[-1]["gini"] < a[-1]["gini"],
      f"{b[-1]['gini']} vs {a[-1]['gini']}")

# ---- determinism ---------------------------------------------------
r1 = Simulation(Config(n_agents=150, n_rounds=40, seed=42)).run()
r2 = Simulation(Config(n_agents=150, n_rounds=40, seed=42)).run()
check("same seed reproduces the run exactly",
      all(approx(x["gini"], y["gini"], 1e-12) for x, y in zip(r1, r2)))
r3 = Simulation(Config(n_agents=150, n_rounds=40, seed=43)).run()
check("different seed gives a different run", not approx(r1[-1]["gini"], r3[-1]["gini"], 1e-12))

# ----------------------------------------------------------------------
print("\n" + "=" * 55)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all checks passed")
