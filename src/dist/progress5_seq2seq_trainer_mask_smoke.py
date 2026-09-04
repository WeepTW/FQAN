#!/usr/bin/env python3
"""CPU smoke for Seq2Seq RetFact-only trainer mask plumbing."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from transformers import Seq2SeqTrainingArguments, default_data_collator
from transformers.modeling_outputs import Seq2SeqLMOutput


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / ".external/FINDER/Retriever Codes"))

from seq2seq_retriever import RetFactMaskSeq2SeqTrainer  # noqa: E402


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        if not ids:
            return ""
        return '{"RetFact":"toy fact","Binding":[{"ObjectName":[],"DataName":"","Position":[{"Begin":[],"End":[]}],"Trend":"None","Num":[],"Text":""}],"Reason":""}'


class ToySeq2SeqModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.probe = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(is_encoder_decoder=True)
        self.generation_config = SimpleNamespace()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError("labels are required for this smoke model")
        batch_size, seq_len = labels.shape
        logits = torch.zeros(
            batch_size,
            seq_len,
            self.vocab_size,
            device=labels.device,
            dtype=torch.float32,
        )
        logits = logits + self.probe
        base_loss = logits.sum() * 0 + torch.tensor(2.0, device=labels.device)
        return Seq2SeqLMOutput(loss=base_loss, logits=logits)


class InspectingTrainer(RetFactMaskSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflow_called = False
        self.mask_seen = False

    def _workflow_schema_loss(self, inputs, outputs, schema_loss_mask, retfact_loss_mask):
        self.workflow_called = True
        self.mask_seen = schema_loss_mask is not None and retfact_loss_mask is not None
        return super()._workflow_schema_loss(inputs, outputs, schema_loss_mask, retfact_loss_mask)


def make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
            "labels": [[1, 2, 3, 4]],
            "schema_loss_mask": [[True, True, False, False]],
            "retfact_loss_mask": [[False, False, True, True]],
        }
    )


def make_trainer(schema_loss_enabled: bool) -> InspectingTrainer:
    args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="fqan_seq2seq_mask_smoke_"),
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to=[],
    )
    return InspectingTrainer(
        model=ToySeq2SeqModel(),
        args=args,
        train_dataset=make_dataset(),
        data_collator=default_data_collator,
        schema_tokenizer=ToyTokenizer(),
        schema_loss_enabled=schema_loss_enabled,
    )


def main() -> None:
    schema_trainer = make_trainer(schema_loss_enabled=True)
    assert schema_trainer.args.remove_unused_columns is False
    schema_batch = next(iter(schema_trainer.get_train_dataloader()))
    assert "schema_loss_mask" in schema_batch
    assert "retfact_loss_mask" in schema_batch
    schema_loss = schema_trainer.compute_loss(
        schema_trainer.model,
        dict(schema_batch),
        return_outputs=False,
    )
    assert schema_trainer.workflow_called
    assert schema_trainer.mask_seen
    assert abs(float(schema_loss.detach().cpu()) - 2.0) > 1e-6

    original_trainer = make_trainer(schema_loss_enabled=False)
    original_batch = next(iter(original_trainer.get_train_dataloader()))
    original_loss = original_trainer.compute_loss(
        original_trainer.model,
        dict(original_batch),
        return_outputs=False,
    )
    assert not original_trainer.workflow_called
    assert abs(float(original_loss.detach().cpu()) - 2.0) < 1e-6

    print(
        json.dumps(
            {
                "status": "ok",
                "remove_unused_columns": schema_trainer.args.remove_unused_columns,
                "schema_masks_preserved": schema_trainer.mask_seen,
                "schema_workflow_called": schema_trainer.workflow_called,
                "schema_loss": float(schema_loss.detach().cpu()),
                "original_workflow_called": original_trainer.workflow_called,
                "original_loss": float(original_loss.detach().cpu()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
