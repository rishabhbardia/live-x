
import unittest
import numpy as np

from geometry.pose import omega_of, look_at
from geometry.costs import PillarCost, DeltaPrior, HeightPrior, LandmarkCost, PrincipalPointPrior, StaticPointCost, InvDepthCost

def check_jacobians(seed=0, tol=1e-6):
    """
        Every analytic block against central differences, at random parameters.
        This is the reason for writing them by hand: a wrong derivative still
        converges to something plausible-looking, so it has to be checked rather
        than trusted.
    """
    sig_base, sig_top = 3.0, 3.0
    sig_q, sig_pp = 2.0, 75.0
    sig_ratio = 8.0
    r_in, pillar_h = 4.55, 1.83
    W, H = 1920, 1080
    rng = np.random.default_rng(seed)
    cases = []
    intr = None
    for _ in range(6):
        intr = np.array([1500 + 800 * rng.random(), 940 + 40 * rng.random(),
                         530 + 20 * rng.random()])
        Cc = np.array([6 * rng.random() - 3, -8 - rng.random(), 1.5 + rng.random()])
        om = omega_of(look_at(Cc)) + 0.05 * rng.standard_normal(3)
        th = np.array([rng.uniform(-1.0, 1.0)])
        dl = np.array([0.1 * rng.standard_normal()])
        Q = np.array([3 * rng.standard_normal(), 3 * rng.standard_normal()])
        obs = np.array([900 + 200 * rng.standard_normal(),
                        500 + 200 * rng.standard_normal()])
        dhb = np.array([0.03 * rng.standard_normal()])
        foc, pp = intr[:1].copy(), intr[1:].copy()
        cases.append((PillarCost(obs, False, sig_base, r_in, pillar_h),
                      [foc, pp, Cc, om, th, dl, dhb]))
        cases.append((PillarCost(obs, True, sig_top, r_in, pillar_h),
                      [foc, pp, Cc, om, th, dl, dhb]))
        cases.append((LandmarkCost(obs, sig_q), [foc, pp, Cc, om, Q]))
        # cases.append((FocalRatioPrior(0.9 + 0.2 * rng.random(), sig_ratio),
        #               [foc, np.array([intr[0] * (0.95 + 0.1 * rng.random())])]))
    cases.append((PrincipalPointPrior(W, H, sig_pp),
                  [np.array([W / 2.0 - 10.0, H / 2.0 + 5.0])]))
    cases.append((DeltaPrior(0.25), [np.array([0.07])]))
    cases.append((HeightPrior(0.05), [np.array([-0.02])]))
    cases.append((StaticPointCost(np.array([2.5,2.3,1.0]), np.array([300, 300]), 0.5),
                  [intr, Cc, om]))
    cases.append((InvDepthCost(np.array([1.3]), np.array([1/1.4]), 0.02),
                      [np.array([0.005, 0.4, 0.943])]))
    

    worst = {}
    for cost, params in cases:
        nres = cost.num_residuals()
        sizes = cost.parameter_block_sizes()
        res = np.zeros(nres)
        jac = [np.zeros(nres * s) for s in sizes]
        cost.Evaluate([p.copy() for p in params], res, jac)
        for b, size in enumerate(sizes):
            ana = jac[b].reshape(nres, size)
            num = np.zeros((nres, size))
            for i in range(size):
                step = 1e-6 * max(1.0, abs(params[b][i]))
                up = [p.copy() for p in params]
                dn = [p.copy() for p in params]
                up[b][i] += step
                dn[b][i] -= step
                r1, r2 = np.zeros(nres), np.zeros(nres)
                cost.Evaluate(up, r1, None)
                cost.Evaluate(dn, r2, None)
                num[:, i] = (r1 - r2) / (2 * step)
            err = np.abs(ana - num).max() / max(1.0, np.abs(num).max())
            key = f"{type(cost).__name__}.block{b}"
            worst[key] = max(worst.get(key, 0.0), float(err))
    return worst, all(v <= tol for v in worst.values())


class TestCameraPose(unittest.TestCase):
    def setUp(self):
        self.worst, self.ok = check_jacobians()

    def test_jacobians(self):
        self.assertEqual(self.ok, True)
        for k, v in sorted(self.worst.items()):
            print(f"  {k:28s} max rel err {v:.2e}")
            self.assertLessEqual(v, 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)