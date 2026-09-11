import logging
from pathlib import Path

import numpy as np
import torch

from .metrics import full_sort_metrics


LOGGER = logging.getLogger("pixelrec.trainer")


class PixelRecTrainer:
    def __init__(self, model, device, learning_rate=5e-4, weight_decay=0.0, betas=(0.9, 0.999)):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, batches = 0.0, 0
        for user_ids, input_ids, answers, negatives in dataloader:
            user_ids = user_ids.to(self.device)
            input_ids = input_ids.to(self.device)
            answers = answers.to(self.device)
            negatives = negatives.to(self.device)
            loss = self.model.calculate_loss(input_ids, answers, negatives, None, user_ids)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader, seen_matrix, topk=20):
        self.model.eval()
        item_embeddings = self.model.get_test_item_emb()
        predictions, answers_all = [], []
        for user_ids, input_ids, answers, _ in dataloader:
            input_ids = input_ids.to(self.device)
            sequence_output = self.model.predict(input_ids)[:, -1, :]
            scores = torch.matmul(sequence_output, item_embeddings.transpose(0, 1))
            scores[:, 0] = -torch.inf
            seen = seen_matrix[user_ids.numpy()].toarray().astype(bool)
            seen = torch.from_numpy(seen).to(scores.device)
            scores.masked_fill_(seen, -torch.inf)
            indices = torch.topk(scores, k=min(topk, scores.size(1) - 1), dim=1).indices
            predictions.append(indices.cpu().numpy())
            answers_all.append(answers.numpy())
        predictions = np.concatenate(predictions, axis=0)
        answers_all = np.concatenate(answers_all, axis=0)
        return full_sort_metrics(answers_all, predictions)


def checkpoint_payload(model, config, epoch, metrics):
    return {
        "format": "PixelRec-v1",
        "dataset": config["dataset"],
        "epoch": int(epoch),
        "metrics": metrics,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }


def save_checkpoint(path, model, config, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, config, epoch, metrics), path)
