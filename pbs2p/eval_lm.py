"""Adapter for the vendored lm-evaluation-harness interface."""

from __future__ import annotations

import torch

from lm_eval.base import BaseLM


class EvalLM(BaseLM):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        device: str = "cuda:0",
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.model = model.to(self._device).eval()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.batch_size_per_gpu = batch_size

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return getattr(
            self.model.config,
            "n_ctx",
            self.model.config.max_position_embeddings,
        )

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _model_call(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).logits

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False,
        )
