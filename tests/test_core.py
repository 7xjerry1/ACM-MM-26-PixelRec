import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from pixelrec.compression import LinearTokenCompressor
from pixelrec.config import load_dataset_config, verify_bundled_data
from pixelrec.data import SequenceDataset, load_catalog, load_sequences, normalize_history, pad_history
from pixelrec.metrics import full_sort_metrics
from pixelrec.model import PixelRec
from pixelrec.preprocessing import preprocess_dataset
from pixelrec.recommendation import score_topk
from pixelrec.trainer import PixelRecTrainer, save_checkpoint
from pixelrec.vlm import PixelRecVLMExtractor


class DatasetTests(unittest.TestCase):
    def test_bundled_datasets_and_counts(self):
        expected = {"beauty": (22363, 12101), "games": (24303, 10672), "toys": (19412, 11924)}
        for dataset, (users, items) in expected.items():
            paths = verify_bundled_data(dataset)
            sequences = load_sequences(paths["sequence"])
            catalog, _ = load_catalog(paths["catalog"])
            self.assertEqual(len(sequences), users)
            self.assertEqual(len(catalog), items)
            self.assertEqual(max(max(row) for row in sequences), items)

    def test_split_and_history_helpers(self):
        sequences = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        train = SequenceDataset(sequences, max_length=3, split="train", item_size=7)
        valid = SequenceDataset(sequences, max_length=3, split="valid", item_size=7)
        test = SequenceDataset(sequences, max_length=3, split="test", item_size=7)
        self.assertEqual(len(train), 6)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(test), 2)
        _, by_asin = load_catalog(verify_bundled_data("beauty")["catalog"])
        first_asin = next(iter(by_asin))
        normalized = normalize_history([first_asin, "2"], by_asin, 12102, max_length=50)
        self.assertEqual(len(pad_history(normalized, 50)), 50)


class PreprocessingTests(unittest.TestCase):
    def test_id_or_asin_images_and_missing_image_fallback(self):
        paths = verify_bundled_data("beauty")
        catalog, _ = load_catalog(paths["catalog"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (24, 36), (20, 40, 60)).save(images / f"{catalog[1]['asin']}.png")
            manifest_path = preprocess_dataset("beauty", images, root / "output", limit=2)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_items"], 2)
            self.assertTrue((root / "output/posters/1.jpg").is_file())
            self.assertTrue((root / "output/posters/2.jpg").is_file())
            with Image.open(root / "output/posters/1.jpg") as image:
                self.assertEqual(image.size, (384, 544))


class FeatureTests(unittest.TestCase):
    def test_vlm_pooling_contract(self):
        hidden = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
        mask = torch.ones(1, 6, dtype=torch.long)
        image_token_id = 99
        input_ids = torch.tensor([[1, 99, 99, 99, 99, 2]])
        grid = torch.tensor([[1, 2, 2]])
        tokens, token_mask, _ = PixelRecVLMExtractor.pool_image_tokens_with_special_tokens(
            hidden, mask, input_ids, grid, image_token_id, 1, 1, 2, 4
        )
        self.assertEqual(tuple(tokens.shape), (1, 4, 3))
        self.assertTrue(token_mask.all())
        self.assertTrue(torch.equal(tokens[0, 0], hidden[0, 0]))
        self.assertTrue(torch.equal(tokens[0, -1], hidden[0, -1]))

    def test_linear_compressor_contract(self):
        model = LinearTokenCompressor(8, 4)
        encoded, reconstructed = model(torch.randn(5, 8))
        self.assertEqual(tuple(encoded.shape), (5, 4))
        self.assertEqual(tuple(reconstructed.shape), (5, 8))
        self.assertIsNone(model.enc.bias)
        self.assertIsNone(model.dec.bias)


class ModelTests(unittest.TestCase):
    def test_pixelrec_forward_loss_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "cache.pt"
            torch.save(
                {"tokens": torch.randn(7, 4, 8), "mask": torch.ones(7, 4, dtype=torch.bool)},
                cache_path,
            )
            args = SimpleNamespace(
                item_size=7, hidden_size=8, max_seq_length=3, batch_size=2,
                initializer_range=0.02, num_attention_heads=2, hidden_act="gelu",
                hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                num_hidden_layers=1, pixelrec_train_stage="transductive_ft",
                pixelrec_temperature=0.07, pixelrec_token_cache_len=4,
                pixelrec_token_dim=8, pixelrec_token_cache_path=str(cache_path),
                pixelrec_token_cache_device="cpu",
                pixelrec_token_compute_dtype="float32", pixelrec_num_queries=2,
                pixelrec_attn_dim=4, pixelrec_dropout=0.0, cuda_condition=False,
            )
            model = PixelRec(args)
            input_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
            output = model(input_ids)
            self.assertEqual(tuple(output.shape), (2, 3, 8))
            loss = model.calculate_loss(input_ids, torch.tensor([3, 4]), None, None, None)
            self.assertTrue(torch.isfinite(loss))
            metrics = full_sort_metrics(np.array([3, 5]), np.array([[3, 1, 2, 4, 5], [1, 2, 3, 4, 5]]))
            self.assertEqual(metrics["H@5"], 1.0)

    def test_tiny_train_checkpoint_evaluate_and_recommend(self):
        from pixelrec.data import make_dataloaders, rating_matrix
        from pixelrec.runtime import load_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "cache.pt"
            mask = torch.ones(7, 4, dtype=torch.bool)
            mask[0] = False
            torch.save({"tokens": torch.randn(7, 4, 8), "mask": mask}, cache_path)
            args = SimpleNamespace(
                item_size=7, hidden_size=8, max_seq_length=3, batch_size=2,
                initializer_range=0.02, num_attention_heads=2, hidden_act="gelu",
                hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                num_hidden_layers=1, pixelrec_train_stage="transductive_ft",
                pixelrec_temperature=0.07, pixelrec_token_cache_len=4,
                pixelrec_token_dim=8, pixelrec_token_cache_path=str(cache_path),
                pixelrec_token_cache_device="cpu", pixelrec_token_compute_dtype="float32",
                pixelrec_num_queries=2, pixelrec_attn_dim=4, pixelrec_dropout=0.0,
                cuda_condition=False,
            )
            sequences = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
            loaders = make_dataloaders(sequences, 7, 2, 3, num_workers=0)
            model = PixelRec(args)
            trainer = PixelRecTrainer(model, torch.device("cpu"), learning_rate=1e-3)
            self.assertTrue(np.isfinite(trainer.train_epoch(loaders["train"])))
            metrics = trainer.evaluate(loaders["test"], rating_matrix(sequences, 7, "test"), topk=5)
            self.assertIn("N@20", metrics)
            checkpoint = Path(temporary) / "best.pt"
            save_checkpoint(checkpoint, model, {"dataset": "tiny"}, 0, metrics)
            reloaded = PixelRec(args)
            load_checkpoint(reloaded, checkpoint, torch.device("cpu"))
            by_id = {item_id: {"item_id": item_id, "asin": f"A{item_id}", "title": f"Item {item_id}"} for item_id in range(1, 7)}
            input_ids = torch.tensor([[0, 1, 2]])
            recommendations = score_topk(reloaded, input_ids, [1, 2], by_id, 3)
            self.assertEqual(len(recommendations), 3)
            self.assertTrue({row["item_id"] for row in recommendations}.isdisjoint({1, 2}))

            legacy_state = {
                name.replace("pixelrec_token_aggregator.", "vlm_token_aggregator."): value
                for name, value in model.state_dict().items()
            }
            legacy_checkpoint = Path(temporary) / "legacy.pt"
            torch.save(legacy_state, legacy_checkpoint)
            legacy_reloaded = PixelRec(args)
            load_checkpoint(legacy_reloaded, legacy_checkpoint, torch.device("cpu"))
            self.assertTrue(
                torch.equal(
                    legacy_reloaded.pixelrec_token_aggregator.query_tokens,
                    model.pixelrec_token_aggregator.query_tokens,
                )
            )


if __name__ == "__main__":
    unittest.main()
