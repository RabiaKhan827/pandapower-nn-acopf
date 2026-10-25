import unittest
import numpy as np
from acopf_surrogate import NeuralACOPFSurrogate

class TestACOPF(unittest.TestCase):
    def test_surrogate_forward_and_bounds(self):
        surrogate = NeuralACOPFSurrogate()
        p = np.array([0.0, 0.4, 0.3, 0.5, 0.2])
        q = np.array([0.0, 0.15, 0.10, 0.20, 0.08])
        v, theta = surrogate.forward(p, q)
        self.assertEqual(len(v), 5)
        self.assertEqual(v[0], 1.00)
        self.assertTrue(np.all(v >= 0.95))
        self.assertTrue(np.all(v <= 1.05))

    def test_benchmark_speedup(self):
        rep = NeuralACOPFSurrogate().benchmark_against_newton_raphson()
        self.assertGreater(rep.inference_speedup_factor, 100.0)
        self.assertEqual(rep.constraint_violation_rate, 0.0)

if __name__ == "__main__":
    unittest.main()
