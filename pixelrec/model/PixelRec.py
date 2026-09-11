import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SequentialRecModel
from .modules import LayerNorm, TransformerEncoder


class PixelRecTokenAggregator(nn.Module):
    """
    Aggregate token-level VLM states with a few learned queries before projection.
    """

    def __init__(self, args, token_dim):
        super().__init__()
        self.hidden_size = int(args.hidden_size)
        self.token_dim = int(token_dim)
        self.num_queries = int(getattr(args, "pixelrec_num_queries", 4))
        self.attn_dim = int(getattr(args, "pixelrec_attn_dim", 256))
        self.dropout_prob = float(getattr(args, "pixelrec_dropout", 0.1))

        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, self.attn_dim) * 0.02)
        self.query_context = nn.Linear(self.hidden_size, self.attn_dim)
        self.key_proj = nn.Linear(self.token_dim, self.attn_dim)
        self.value_norm = nn.LayerNorm(self.token_dim)
        self.attn_dropout = nn.Dropout(self.dropout_prob)
        self.query_score = nn.Linear(self.token_dim, 1)

        projector_hidden = max(self.hidden_size * 4, self.hidden_size)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, projector_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(projector_hidden, self.hidden_size),
        )

    def forward(self, token_states, token_mask, context_emb=None, return_details=False):
        queries = self.query_tokens.unsqueeze(0).expand(token_states.size(0), -1, -1)
        if context_emb is not None:
            queries = queries + self.query_context(context_emb).unsqueeze(1)

        keys = self.key_proj(token_states)
        values = self.value_norm(token_states)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        scores = scores.masked_fill(~token_mask.unsqueeze(1), -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = attn * token_mask.unsqueeze(1).to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn_for_output = self.attn_dropout(attn)

        attended = torch.matmul(attn_for_output, values)
        query_weights = F.softmax(self.query_score(attended).squeeze(-1), dim=-1)
        pooled = (query_weights.unsqueeze(-1) * attended).sum(dim=1)
        projected = self.projector(pooled)

        if not return_details:
            return projected

        effective_token_weights = (query_weights.unsqueeze(-1) * attn).sum(dim=1)
        return projected, {
            "attn": attn,
            "query_weights": query_weights,
            "effective_token_weights": effective_token_weights,
        }


class PixelRec(SequentialRecModel):
    """
    PixelRec: token-level VLM cache with a lightweight query aggregator.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "PixelRec"
        self.train_stage = args.pixelrec_train_stage
        self.temperature = args.pixelrec_temperature
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TransformerEncoder(args)
        self.eval_item_chunk_size = 512

        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self.init_weights)
        self._loss_stats = {}
        self._cached_eval_item_emb = None

        self.token_cache_len = int(getattr(args, "pixelrec_token_cache_len", 0))
        configured_token_dim = getattr(args, "pixelrec_token_dim", None)
        if configured_token_dim is not None:
            configured_token_dim = int(configured_token_dim)
        self.vlm_token_cache_device = str(
            getattr(args, "pixelrec_token_cache_device", "gpu")
        ).lower()
        self.vlm_token_compute_dtype_name = str(
            getattr(args, "pixelrec_token_compute_dtype", "float32")
        ).lower()
        self.vlm_token_compute_dtype = self._resolve_token_compute_dtype(
            self.vlm_token_compute_dtype_name
        )
        if self.vlm_token_compute_dtype != torch.float32 and not bool(
            getattr(args, "cuda_condition", torch.cuda.is_available())
        ):
            raise ValueError(
                "pixelrec_token_compute_dtype must be float32 when training on CPU. "
                "Half-precision token aggregation requires CUDA."
            )
        token_states, token_mask = self._load_vlm_token_cache(
            cache_path=args.pixelrec_token_cache_path,
            item_size=args.item_size,
            token_dim=configured_token_dim,
        )
        self.token_dim = int(token_states.size(-1))
        if self.vlm_token_cache_device == "gpu":
            self.register_buffer("vlm_token_states", token_states, persistent=False)
            self.register_buffer("vlm_token_mask", token_mask, persistent=False)
            self._vlm_token_states_cpu = None
            self._vlm_token_mask_cpu = None
        elif self.vlm_token_cache_device == "cpu":
            self.vlm_token_states = None
            self.vlm_token_mask = None
            self._vlm_token_states_cpu = token_states
            self._vlm_token_mask_cpu = token_mask
        else:
            raise ValueError(
                f"unsupported pixelrec_token_cache_device={self.vlm_token_cache_device!r}; expected 'gpu' or 'cpu'."
            )
        self.pixelrec_token_aggregator = PixelRecTokenAggregator(args, self.token_dim)

    def train(self, mode=True):
        self._cached_eval_item_emb = None
        return super().train(mode)

    def _load_vlm_token_cache(self, cache_path, item_size, token_dim=None):
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict):
            token_states = payload.get("tokens")
            token_mask = payload.get("mask")
        elif isinstance(payload, (list, tuple)) and len(payload) == 2:
            token_states, token_mask = payload
        else:
            raise ValueError(
                "vlm token cache must be a dict with `tokens`/`mask` or a 2-tuple `(tokens, mask)`."
            )

        if not isinstance(token_states, torch.Tensor):
            raise ValueError(f"vlm token cache `tokens` must be a Tensor, got: {type(token_states)}")
        if not isinstance(token_mask, torch.Tensor):
            raise ValueError(f"vlm token cache `mask` must be a Tensor, got: {type(token_mask)}")
        if token_states.ndim != 3:
            raise ValueError(
                f"vlm token cache `tokens` must be 3D Tensor [item_size, max_tokens, dim], got shape={tuple(token_states.shape)}"
            )
        if token_mask.ndim != 2:
            raise ValueError(
                f"vlm token cache `mask` must be 2D Tensor [item_size, max_tokens], got shape={tuple(token_mask.shape)}"
            )
        if token_states.size(0) != item_size:
            raise ValueError(
                f"vlm token cache item_size ({token_states.size(0)}) != dataset item_size ({item_size})."
            )
        if token_dim is not None and token_states.size(2) != token_dim:
            raise ValueError(
                f"vlm token cache hidden dim ({token_states.size(2)}) != expected ({token_dim})."
            )
        if token_mask.size(0) != item_size or token_mask.size(1) != token_states.size(1):
            raise ValueError(
                f"vlm token cache mask shape {tuple(token_mask.shape)} is incompatible with token shape {tuple(token_states.shape)}."
            )
        return token_states.contiguous(), token_mask.bool().contiguous()

    @staticmethod
    def _resolve_token_compute_dtype(dtype_name):
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in mapping:
            raise ValueError(
                f"unsupported pixelrec_token_compute_dtype={dtype_name!r}; "
                "expected 'float32', 'float16', or 'bfloat16'."
            )
        return mapping[dtype_name]

    def _compute_dtype(self):
        return self.item_embeddings.weight.dtype

    def _gather_vlm_token_cache(self, item_ids):
        if self.vlm_token_cache_device == "gpu":
            return self.vlm_token_states[item_ids], self.vlm_token_mask[item_ids]

        item_ids_cpu = item_ids.detach().to(device="cpu")
        token_states = self._vlm_token_states_cpu[item_ids_cpu]
        token_mask = self._vlm_token_mask_cpu[item_ids_cpu]
        target_device = item_ids.device
        non_blocking = target_device.type == "cuda"
        token_states = token_states.to(device=target_device, non_blocking=non_blocking)
        token_mask = token_mask.to(device=target_device, non_blocking=non_blocking)
        return token_states, token_mask

    def _get_token_context_emb(self, id_emb):
        if self.train_stage == "transductive_ft":
            return id_emb
        return None

    def _apply_invalid_item_mask(self, item_emb, item_ids, token_mask):
        invalid_mask = item_ids.eq(0) | (~token_mask.any(dim=-1))
        if invalid_mask.any():
            item_emb = item_emb.masked_fill(invalid_mask.unsqueeze(-1), 0.0)
        return item_emb

    def _apply_id_residual(self, item_emb, id_emb):
        if self.train_stage == "transductive_ft":
            item_emb = item_emb + id_emb
        return item_emb

    def _encode_vlm_tokens(self, token_states, token_mask, context_emb=None, aggregator=None):
        leading_shape = token_states.shape[:-2]
        token_states = token_states.reshape(-1, token_states.size(-2), self.token_dim)
        token_mask = token_mask.reshape(-1, token_mask.size(-1))
        model_dtype = self._compute_dtype()
        aggregator = self.pixelrec_token_aggregator if aggregator is None else aggregator

        if self.vlm_token_compute_dtype == torch.float32:
            token_states = token_states.to(dtype=model_dtype)
            if context_emb is not None:
                context_emb = context_emb.reshape(-1, self.args.hidden_size).to(dtype=model_dtype)
            encoded = aggregator(token_states, token_mask, context_emb=context_emb)
            return encoded.view(*leading_shape, self.args.hidden_size)

        target_dtype = self.vlm_token_compute_dtype
        if token_states.device.type != "cuda":
            raise ValueError(
                "Half-precision token aggregation requires CUDA, "
                f"but got token_states on device={token_states.device}."
            )

        token_states = token_states.to(dtype=target_dtype)
        if context_emb is not None:
            context_emb = context_emb.reshape(-1, self.args.hidden_size).to(dtype=target_dtype)
        with torch.autocast(device_type="cuda", dtype=target_dtype):
            encoded = aggregator(token_states, token_mask, context_emb=context_emb)
        encoded = encoded.to(dtype=model_dtype)
        return encoded.view(*leading_shape, self.args.hidden_size)

    def _analyze_vlm_tokens(self, token_states, token_mask, context_emb=None, aggregator=None):
        leading_shape = token_states.shape[:-2]
        token_states = token_states.reshape(-1, token_states.size(-2), self.token_dim)
        token_mask = token_mask.reshape(-1, token_mask.size(-1))
        model_dtype = self._compute_dtype()
        aggregator = self.pixelrec_token_aggregator if aggregator is None else aggregator

        if self.vlm_token_compute_dtype == torch.float32:
            token_states = token_states.to(dtype=model_dtype)
            if context_emb is not None:
                context_emb = context_emb.reshape(-1, self.args.hidden_size).to(dtype=model_dtype)
            encoded, details = aggregator(
                token_states,
                token_mask,
                context_emb=context_emb,
                return_details=True,
            )
        else:
            target_dtype = self.vlm_token_compute_dtype
            if token_states.device.type != "cuda":
                raise ValueError(
                    "Half-precision token aggregation requires CUDA, "
                    f"but got token_states on device={token_states.device}."
                )

            token_states = token_states.to(dtype=target_dtype)
            if context_emb is not None:
                context_emb = context_emb.reshape(-1, self.args.hidden_size).to(dtype=target_dtype)
            with torch.autocast(device_type="cuda", dtype=target_dtype):
                encoded, details = aggregator(
                    token_states,
                    token_mask,
                    context_emb=context_emb,
                    return_details=True,
                )

        encoded = encoded.to(dtype=model_dtype).view(*leading_shape, self.args.hidden_size)
        reshaped_details = {}
        for key, value in details.items():
            reshaped_details[key] = value.to(dtype=torch.float32).view(*leading_shape, *value.shape[1:])
        return encoded, reshaped_details

    def analyze_item_token_aggregation(self, item_ids, token_states=None, token_mask=None, id_emb=None):
        if token_states is None or token_mask is None:
            token_states, token_mask = self._gather_vlm_token_cache(item_ids)
        if id_emb is None:
            id_emb = self.item_embeddings(item_ids)

        context_emb = self._get_token_context_emb(id_emb)
        item_emb, details = self._analyze_vlm_tokens(
            token_states,
            token_mask,
            context_emb=context_emb,
        )
        item_emb = self._apply_invalid_item_mask(item_emb, item_ids, token_mask)
        item_emb_with_id = self._apply_id_residual(item_emb, id_emb)
        details.update(
            {
                "item_emb": item_emb,
                "item_emb_with_id": item_emb_with_id,
                "token_mask": token_mask,
                "item_ids": item_ids,
            }
        )
        return details

    def _encode_item_ids(self, item_ids, id_emb=None):
        token_states, token_mask = self._gather_vlm_token_cache(item_ids)
        context_emb = self._get_token_context_emb(id_emb)
        item_emb = self._encode_vlm_tokens(token_states, token_mask, context_emb=context_emb)
        return self._apply_invalid_item_mask(item_emb, item_ids, token_mask)

    def _build_multimodal_item_emb(self, item_seq=None, for_test=False):
        if for_test:
            item_ids = torch.arange(self.args.item_size, device=self.item_embeddings.weight.device, dtype=torch.long)
            id_emb = self.item_embeddings(item_ids)
        else:
            item_ids = item_seq
            id_emb = self.item_embeddings(item_seq)

        item_emb = self._encode_item_ids(item_ids, id_emb=id_emb)
        return self._apply_id_residual(item_emb, id_emb)

    def _build_sequence_input(self, item_seq):
        item_emb = self._build_multimodal_item_emb(item_seq=item_seq, for_test=False)

        seq_length = item_seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embeddings(position_ids)

        input_emb = self.LayerNorm(item_emb + position_embedding)
        return self.dropout(input_emb)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self._build_sequence_input(input_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        if all_sequence_output:
            return item_encoded_layers
        return item_encoded_layers[-1]

    def get_test_item_emb(self):
        if (not self.training) and self._cached_eval_item_emb is not None:
            return self._cached_eval_item_emb

        device = self.item_embeddings.weight.device
        chunks = []
        context = torch.enable_grad() if self.training else torch.no_grad()
        with context:
            for start in range(0, self.args.item_size, self.eval_item_chunk_size):
                end = min(self.args.item_size, start + self.eval_item_chunk_size)
                item_ids = torch.arange(start, end, device=device, dtype=torch.long)
                id_emb = self.item_embeddings(item_ids)
                item_emb = self._encode_item_ids(item_ids, id_emb=id_emb)
                item_emb = self._apply_id_residual(item_emb, id_emb)
                chunks.append(item_emb)
            test_item_emb = torch.cat(chunks, dim=0)
            test_item_emb = F.normalize(test_item_emb, dim=-1)

        if not self.training:
            self._cached_eval_item_emb = test_item_emb
        return test_item_emb

    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        seq_output = self.forward(input_ids, user_ids=user_ids, all_sequence_output=all_sequence_output)
        if isinstance(seq_output, list):
            seq_output = seq_output[-1]
        return F.normalize(seq_output, dim=-1)

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = self.get_test_item_emb()
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        rec_loss = self.loss_fct(logits, answers)

        self._loss_stats = {
            "rec_loss": float(rec_loss.detach().item()),
            "total_loss": float(rec_loss.detach().item()),
        }
        return rec_loss
