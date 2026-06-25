from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ast_preprocessor import ASTFeatureExtractor, ASTPreprocessConfig, TypeVocabulary
from inference_ast import IncrementalASTBuilder, IncrementalASTConfig
from scripts.prefix_ast_degradation import measure as measure_prefix_ast_degradation

try:
    import torch

    from csmt_gnn import ASTGatedPool, BoundaryAwarePoolInput, CSMTConfig, CSMTModel, PrefixBlockGraph
    from transformer_baseline import TinyCausalTransformer, TransformerBaselineConfig
except ImportError:  # pragma: no cover - lets syntax-only environments still run unittest discovery.
    torch = None
    ASTGatedPool = BoundaryAwarePoolInput = CSMTConfig = CSMTModel = PrefixBlockGraph = None
    TinyCausalTransformer = TransformerBaselineConfig = None


class ASTVocabularyTests(unittest.TestCase):
    def test_frozen_ast_vocab_maps_unseen_types_to_unknown(self) -> None:
        vocab = TypeVocabulary({"<PAD>": 0, "<UNKNOWN>": 1, "identifier": 2}, frozen=True)
        self.assertEqual(vocab.id_for("identifier"), 2)
        self.assertEqual(vocab.id_for("never_seen_node"), 1)
        self.assertEqual(len(vocab.type_to_id), 3)

    def test_incremental_ast_reports_fallback_and_unknown_rate_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_path = Path(temp_dir) / "ast_vocab.json"
            vocab_path.write_text(
                json.dumps({"type_vocab": {"<PAD>": 0, "<UNKNOWN>": 1, "identifier": 2}}),
                encoding="utf-8",
            )
            builder = IncrementalASTBuilder(IncrementalASTConfig(block_size=8, max_tokens=32, vocab_path=vocab_path))
            result = builder.build("def f(x:\n    return x +")
            self.assertGreater(result.ast_ids.size, 0)
            self.assertGreaterEqual(result.fallback_rate, 0.0)
            self.assertLessEqual(result.fallback_rate, 1.0)
            self.assertIn(1, result.ast_ids)

    def test_incremental_ast_config_validates_shape_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_size must be positive"):
            IncrementalASTConfig(block_size=0)
        with self.assertRaisesRegex(TypeError, "max_tokens must be an integer"):
            IncrementalASTConfig(max_tokens=32.0)

    def test_ast_extractor_reports_token_fallback_mask(self) -> None:
        extractor = ASTFeatureExtractor(ASTPreprocessConfig(block_size=4, max_tokens=32), TypeVocabulary())
        extractor.extract("def f(x:\n    return x +")
        fallback_mask = extractor.last_stats.get("fallback_token_mask")
        self.assertIsInstance(fallback_mask, list)
        self.assertEqual(len(fallback_mask), extractor.last_stats["num_tokens"])

    def test_prefix_ast_degradation_measurement_has_bounded_rates(self) -> None:
        result = measure_prefix_ast_degradation(
            "def f(x):\n    y = x + 1\n    return y\n",
            block_size=4,
            max_tokens=32,
            vocab_path=None,
            tokenizer_name_or_path=None,
            repeat=1,
        )
        for key in ("fallback_rate", "unknown_rate", "prefix_full_divergence"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)
        self.assertGreater(result["num_tokens"], 0)


@unittest.skipIf(torch is None, "PyTorch is not installed in this Python environment.")
class CSMTShapeTests(unittest.TestCase):
    def tiny_config(self) -> CSMTConfig:
        return CSMTConfig(
            vocab_size=64,
            num_layers=1,
            hidden_size=16,
            block_size=4,
            max_tokens=16,
            num_heads=4,
            num_graph_heads=4,
            num_experts=1,
            moe_top_k=1,
            ast_dim=8,
            num_ast_types=16,
            boundary_mix=0.1,
            cvd_prob=0.0,
        )

    def test_pool_handles_empty_padding_blocks(self) -> None:
        config = self.tiny_config()
        pool = ASTGatedPool(config)
        h = torch.randn(1, 3, config.block_size, config.hidden_size)
        token_mask = torch.zeros(1, 3, config.block_size, dtype=torch.bool)
        token_mask[:, 0, :] = True
        _, z = pool(h, ast_embeds=None, token_mask=token_mask)
        self.assertFalse(torch.isnan(z).any())
        self.assertTrue(torch.allclose(z[:, 1:], torch.zeros_like(z[:, 1:])))

    def test_boundary_mixing_is_causal(self) -> None:
        config = self.tiny_config()
        module = BoundaryAwarePoolInput(config)
        with torch.no_grad():
            self.assertIsNotNone(module.left_proj)
            module.left_proj.weight.copy_(torch.eye(config.hidden_size))
        h = torch.zeros(1, 2, config.block_size, config.hidden_size)
        h[:, 0, -1, :] = 1.0
        h[:, 1, 0, :] = 3.0
        token_mask = torch.ones(1, 2, config.block_size, dtype=torch.bool)
        mixed = module(h, token_mask)
        self.assertTrue(torch.all(mixed[:, 1, 0, :] > h[:, 1, 0, :]))
        self.assertTrue(torch.allclose(mixed[:, 0, -1, :], h[:, 0, -1, :]))

    def test_boundary_width_updates_multiple_head_tokens(self) -> None:
        config = CSMTConfig(**{**self.tiny_config().__dict__, "boundary_width": 2})
        module = BoundaryAwarePoolInput(config)
        with torch.no_grad():
            self.assertIsNotNone(module.left_proj)
            module.left_proj.weight.copy_(torch.cat([torch.eye(config.hidden_size), torch.eye(config.hidden_size)], dim=0))
        h = torch.zeros(1, 2, config.block_size, config.hidden_size)
        h[:, 0, -1, :] = 1.0
        token_mask = torch.ones(1, 2, config.block_size, dtype=torch.bool)
        mixed = module(h, token_mask)
        self.assertTrue(torch.all(mixed[:, 1, :2, :] > h[:, 1, :2, :]))
        self.assertTrue(torch.allclose(mixed[:, 1, 2:, :], h[:, 1, 2:, :]))

    def test_model_accepts_true_lengths(self) -> None:
        config = self.tiny_config()
        model = CSMTModel(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
        lengths = torch.tensor([4, 6])
        ast = torch.zeros(2, 2, config.block_size, dtype=torch.long)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        logits = model(input_ids, ast_type_ids=ast, var_def_mask=mask, lengths=lengths)
        self.assertEqual(tuple(logits.shape), (2, 6, config.vocab_size))

    def test_shared_ast_and_mask_inputs_broadcast_across_batch(self) -> None:
        config = self.tiny_config()
        model = CSMTModel(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        ast = torch.zeros(2, config.block_size, dtype=torch.long)
        token_mask = torch.zeros(6, dtype=torch.bool)
        logits = model(input_ids, ast_type_ids=ast, var_def_mask=token_mask, lengths=torch.tensor([6, 6]))
        self.assertEqual(tuple(logits.shape), (2, 6, config.vocab_size))

    def test_invalid_config_values_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_size must be divisible"):
            CSMTConfig(**{**self.tiny_config().__dict__, "hidden_size": 18, "num_heads": 4})
        with self.assertRaisesRegex(ValueError, "kv_compression must be in"):
            CSMTConfig(**{**self.tiny_config().__dict__, "kv_compression": 1.5})
        with self.assertRaisesRegex(TypeError, "use_ast_gate must be a bool"):
            CSMTConfig(**{**self.tiny_config().__dict__, "use_ast_gate": 1})

    def test_invalid_forward_inputs_fail_early(self) -> None:
        config = self.tiny_config()
        model = CSMTModel(config)
        with self.assertRaisesRegex(TypeError, "input_ids must contain"):
            model(torch.tensor([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "input_ids ids must be"):
            model(torch.tensor([[1, config.vocab_size]]))
        with self.assertRaisesRegex(TypeError, "lengths must contain"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([3.0]))
        with self.assertRaisesRegex(ValueError, "lengths must be in"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([4]))
        with self.assertRaisesRegex(TypeError, "ast_type_ids must contain"):
            model(torch.tensor([[1, 2, 3]]), ast_type_ids=torch.zeros(1, config.block_size))
        with self.assertRaisesRegex(ValueError, "ast_type_ids ids must be"):
            model(
                torch.tensor([[1, 2, 3]]),
                ast_type_ids=torch.full((1, config.block_size), config.num_ast_types, dtype=torch.long),
            )
        with self.assertRaisesRegex(TypeError, "var_def_mask must be"):
            model(torch.tensor([[1, 2, 3]]), var_def_mask=torch.zeros(3, dtype=torch.float32))

    def test_input_range_validation_can_be_disabled_after_pipeline_checks(self) -> None:
        config = CSMTConfig(**{**self.tiny_config().__dict__, "validate_input_ranges": False})
        model = CSMTModel(config)
        logits = model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([3]))
        self.assertEqual(tuple(logits.shape), (1, 3, config.vocab_size))
        logits = model(torch.tensor([[1, 2, 3]]), ast_type_ids=torch.zeros(1, config.block_size, dtype=torch.long), lengths=torch.tensor([3]))
        self.assertEqual(tuple(logits.shape), (1, 3, config.vocab_size))
        with self.assertRaisesRegex(TypeError, "input_ids must contain"):
            model(torch.tensor([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "lengths must be in"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([4]))

    def test_ablation_toggles_preserve_forward_shape(self) -> None:
        toggle_sets = [
            {"use_ast_gate": False},
            {"use_block_graph": False},
            {"use_cvd": False},
            {"use_moe": False},
            {"use_boundary": False},
            {
                "use_ast_gate": False,
                "use_block_graph": False,
                "use_cvd": False,
                "use_moe": False,
                "use_boundary": False,
            },
        ]
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        ast = torch.zeros(1, 2, self.tiny_config().block_size, dtype=torch.long)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        for toggles in toggle_sets:
            config = CSMTConfig(**{**self.tiny_config().__dict__, **toggles})
            model = CSMTModel(config)
            logits = model(input_ids, ast_type_ids=ast, var_def_mask=mask, lengths=torch.tensor([6]))
            self.assertEqual(tuple(logits.shape), (1, 6, config.vocab_size))

    def test_random_cvd_samples_only_valid_blocks(self) -> None:
        config = CSMTConfig(**{**self.tiny_config().__dict__, "cvd_prob": 1.0, "cvd_scope": "random"})
        graph = PrefixBlockGraph(config)
        graph.train()
        z = torch.randn(1, 3, config.hidden_size)
        valid = torch.tensor([[True, True, False]])
        _, sampled = graph(z, var_def_mask=None, valid_block_mask=valid)
        self.assertIsNotNone(sampled)
        self.assertTrue(torch.equal(sampled, valid))

    def test_variable_cvd_ignores_padding_blocks(self) -> None:
        config = CSMTConfig(**{**self.tiny_config().__dict__, "cvd_prob": 1.0, "cvd_audit": True})
        graph = PrefixBlockGraph(config)
        graph.train()
        z = torch.randn(1, 3, config.hidden_size)
        var_defs = torch.tensor([[False, True, True]])
        valid = torch.tensor([[True, True, False]])
        _, sampled = graph(z, var_def_mask=var_defs, valid_block_mask=valid)
        self.assertIsNotNone(sampled)
        self.assertTrue(torch.equal(sampled, torch.tensor([[False, True, False]])))
        self.assertEqual(graph.last_cvd_audit["eligible_blocks"], 1.0)

    def test_cvd_audit_is_opt_in(self) -> None:
        config = CSMTConfig(**{**self.tiny_config().__dict__, "cvd_prob": 1.0})
        graph = PrefixBlockGraph(config)
        graph.train()
        z = torch.randn(1, 2, config.hidden_size)
        var_defs = torch.tensor([[True, False]])
        valid = torch.tensor([[True, True]])
        _, sampled = graph(z, var_def_mask=var_defs, valid_block_mask=valid)
        self.assertIsNotNone(sampled)
        self.assertEqual(graph.last_cvd_audit, {})

    def test_no_moe_uses_dense_ffn(self) -> None:
        config = CSMTConfig(**{**self.tiny_config().__dict__, "use_moe": False})
        model = CSMTModel(config)
        layer = model.layers[0]
        self.assertIsNone(layer.moe)
        self.assertIsNotNone(layer.dense_ffn)
        self.assertEqual(float(model.moe_auxiliary_loss().item()), 0.0)

    def test_tiny_transformer_forward_shape(self) -> None:
        config = TransformerBaselineConfig(
            vocab_size=64,
            num_layers=1,
            hidden_size=16,
            max_tokens=16,
            num_heads=4,
            ffn_multiplier=1.5,
        )
        model = TinyCausalTransformer(config)
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        logits = model(input_ids, lengths=torch.tensor([3, 2]))
        self.assertEqual(tuple(logits.shape), (2, 4, config.vocab_size))

    def test_tiny_transformer_rejects_invalid_inputs(self) -> None:
        config = TransformerBaselineConfig(
            vocab_size=64,
            num_layers=1,
            hidden_size=16,
            max_tokens=16,
            num_heads=4,
            ffn_multiplier=1.5,
        )
        model = TinyCausalTransformer(config)
        with self.assertRaisesRegex(TypeError, "input_ids must contain"):
            model(torch.tensor([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "input_ids ids must be"):
            model(torch.tensor([[1, config.vocab_size]]))
        with self.assertRaisesRegex(TypeError, "lengths must contain"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([3.0]))
        with self.assertRaisesRegex(ValueError, "lengths must be in"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([4]))

    def test_tiny_transformer_config_validates_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "hidden_size must be an integer"):
            TransformerBaselineConfig(hidden_size=16.0)
        with self.assertRaisesRegex(TypeError, "tie_embeddings must be a bool"):
            TransformerBaselineConfig(tie_embeddings=1)
        with self.assertRaisesRegex(TypeError, "validate_input_ranges must be a bool"):
            TransformerBaselineConfig(validate_input_ranges=1)

    def test_tiny_transformer_range_validation_can_be_disabled(self) -> None:
        config = TransformerBaselineConfig(
            vocab_size=64,
            num_layers=1,
            hidden_size=16,
            max_tokens=16,
            num_heads=4,
            ffn_multiplier=1.5,
            validate_input_ranges=False,
        )
        model = TinyCausalTransformer(config)
        logits = model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([3]))
        self.assertEqual(tuple(logits.shape), (1, 3, config.vocab_size))
        with self.assertRaisesRegex(TypeError, "lengths must contain"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([3.0]))
        with self.assertRaisesRegex(ValueError, "lengths must be in"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([4]))


    def test_tiny_transformer_is_causal(self) -> None:
        config = TransformerBaselineConfig(
            vocab_size=64,
            num_layers=1,
            hidden_size=16,
            max_tokens=16,
            num_heads=4,
            ffn_multiplier=1.5,
        )
        model = TinyCausalTransformer(config)
        model.eval()
        left = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 9, 10]])
        with torch.no_grad():
            logits_left = model(left)
            logits_changed = model(changed_future)
        self.assertTrue(torch.allclose(logits_left[:, :2], logits_changed[:, :2], atol=1e-5))

    def test_parameter_neighbor_transformer_is_larger(self) -> None:
        base = TinyCausalTransformer(
            TransformerBaselineConfig(
                vocab_size=64,
                num_layers=1,
                hidden_size=16,
                max_tokens=16,
                num_heads=4,
                ffn_multiplier=1.5,
            )
        )
        matched = TinyCausalTransformer(
            TransformerBaselineConfig(
                vocab_size=64,
                num_layers=1,
                hidden_size=16,
                max_tokens=16,
                num_heads=4,
                ffn_multiplier=4.5,
            )
        )
        base_params = sum(param.numel() for param in base.parameters())
        matched_params = sum(param.numel() for param in matched.parameters())
        self.assertGreater(matched_params, base_params)


if __name__ == "__main__":
    unittest.main()
