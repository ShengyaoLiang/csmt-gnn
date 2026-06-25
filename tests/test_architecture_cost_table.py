from __future__ import annotations

import unittest

from scripts.architecture_cost_table import block_local_edges, causal_edges, cost_row


class ArchitectureCostTableTests(unittest.TestCase):
    def test_causal_edges(self) -> None:
        self.assertEqual(causal_edges(1), 1)
        self.assertEqual(causal_edges(4), 10)

    def test_block_local_edges_handles_tail_block(self) -> None:
        self.assertEqual(block_local_edges(sequence_length=10, block_size=4), 10 + 10 + 3)

    def test_csmt_edge_count_is_below_dense_for_long_sequence(self) -> None:
        row = cost_row(sequence_length=1024, block_size=64)
        self.assertEqual(row.num_blocks, 16)
        self.assertLess(row.csmt_edges, row.dense_causal_edges)
        self.assertLess(row.edge_ratio_vs_dense, 0.1)

    def test_continuous_optimum_matches_formula(self) -> None:
        row = cost_row(sequence_length=2048, block_size=64, a=1.0, b=1.0)
        self.assertAlmostEqual(row.continuous_optimal_block, (2.0 * 2048) ** (1.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
