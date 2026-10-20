"""Enumerated checks of the proposed BRPO identities; no RL training."""
import itertools
import json
import math
from pathlib import Path


def kl(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q) if x)


def close(x, y, tol=1e-9):
    assert abs(x - y) < tol, (x, y)


results = {}

# A nonlinear mixer can represent XOR, whose full maximizer set is not Cartesian.
def mixer(x, y):
    return x + y - 2 * x * y

xor = {(a, b): mixer(a, b) for a, b in itertools.product([0, 1], repeat=2)}
assert xor == {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
results["nonmonotonic_xor"] = {"values": list(xor.values()), "dq_du1_at_u2": [1, -1]}

# Fixed context: pi = pi_1 pi_2; mu is genuinely correlated and strictly positive.
mu = [[0.4, 0.1], [0.15, 0.35]]
p1, p2 = [0.3, 0.7], [0.6, 0.4]
joint_p = [p1[a] * p2[b] for a, b in itertools.product(range(2), repeat=2)]
joint_mu = [mu[a][b] for a, b in itertools.product(range(2), repeat=2)]
mu2 = [sum(mu[a][b] for a in range(2)) for b in range(2)]
conditional_term = sum(
    p2[b] * kl(p1, [mu[a][b] / mu2[b] for a in range(2)]) for b in range(2)
)
lhs, rhs = kl(joint_p, joint_mu), conditional_term + kl(p2, mu2)
close(lhs, rhs)
results["joint_kl_coordinate_identity"] = {"joint": lhs, "decomposed": rhs}

# Rooted tree 0 -> 1 -> 2 defines a normalized joint density/mass.
root = [0.4, 0.6]
c1 = [[0.8, 0.2], [0.1, 0.9]]
c2 = [[0.7, 0.3], [0.25, 0.75]]
tree_mass = sum(
    root[a] * c1[a][b] * c2[b][c]
    for a, b, c in itertools.product(range(2), repeat=3)
)
close(tree_mass, 1.0)
results["tree_normalization"] = tree_mass

# An action also appears as the conditioning input of its child density.
# a1~N(0,1), a2|a1~N(rho*a1, sigma^2).
rho, variance, a1, a2 = 0.6, 0.5, 0.2, -0.4
def log_density(x, y):
    return -0.5 * x*x - 0.5 * (y-rho*x)**2 / variance
eps = 1e-6
numeric = (log_density(a1+eps, a2)-log_density(a1-eps, a2))/(2*eps)
full = -a1 + rho * (a2-rho*a1)/variance
close(numeric, full)
assert abs(full - (-a1)) > 0.1
results["full_gradient_includes_child"] = {
    "finite_difference": numeric, "full": full, "own_factor_only": -a1,
}

# Centering the same regularized score yields zero conditional mean credit.
lam = 0.2
scores = [[xor[a, b] + lam * math.log(mu[a][b]) for b in range(2)] for a in range(2)]
means = []
for b in range(2):
    baseline = sum(p1[a] * scores[a][b] for a in range(2))
    means.append(sum(p1[a] * (scores[a][b]-baseline) for a in range(2)))
    close(means[-1], 0.0)
results["regularized_counterfactual_centering"] = means

# Three-way parity cannot be recovered from independent or pairwise marginals.
parity = {a: (0.25 if sum(a) % 2 == 0 else 0.0)
          for a in itertools.product(range(2), repeat=3)}
for i, j in itertools.combinations(range(3), 2):
    for x, y in itertools.product(range(2), repeat=2):
        mass = sum(p for a, p in parity.items() if a[i] == x and a[j] == y)
        close(mass, 0.25)
tv = 0.5 * sum(abs(p - 0.125) for p in parity.values())
close(tv, 0.5)
results["pairwise_limit_parity"] = {"tree_from_true_pair_marginals_tv": tv}

out = Path(__file__).with_name("mathematical_checks.json")
out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
