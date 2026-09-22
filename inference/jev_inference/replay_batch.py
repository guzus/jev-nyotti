"""Cross-symbol batching, gated on actual first-batch sequential parity.

Does not modify the engine or public inference path. Construct once per replay.
A failed gate permanently selects the existing serial implementation.
"""
from __future__ import annotations

import math


def probabilities(logits):
    largest = max(logits)
    values = [math.exp(value-largest) for value in logits]
    total = sum(values)
    return [value/total for value in values]


def equivalent(left, right, tolerance=0.01):
    if len(left) != len(right):
        return False
    for serial, batched in zip(left, right):
        if len(serial.logits) != len(batched.logits):
            return False
        if not all(math.isfinite(v) for v in serial.logits+batched.logits):
            return False
        if max(range(len(serial.logits)),key=lambda i:serial.logits[i]) != max(range(len(batched.logits)),key=lambda i:batched.logits[i]):
            return False
        if max(abs(a-b) for a,b in zip(probabilities(serial.logits),probabilities(batched.logits))) > tolerance:
            return False
    return True


class ReplayBatchScorer:
    def __init__(self):
        self.enabled = None
        self.gate = 'not_checked'

    def _batch(self, engine, prepared):
        from .schemas import JobScore
        import torch
        width = max(len(job.input_ids) for job in prepared)
        pad = engine.tokenizer.pad_token_id
        if pad is None:
            pad = engine.tokenizer.eos_token_id
        if not isinstance(pad,int):
            raise ValueError('missing scalar padding token')
        inputs = torch.tensor([[pad]*(width-len(job.input_ids))+job.input_ids for job in prepared],dtype=torch.long,device=engine.settings.device)
        mask = torch.tensor([[0]*(width-len(job.input_ids))+[1]*len(job.input_ids) for job in prepared],dtype=torch.long,device=engine.settings.device)
        with torch.inference_mode():
            output = engine.model(input_ids=inputs,attention_mask=mask,use_cache=False,logits_to_keep=1,return_dict=True)
            result=[]
            for i,job in enumerate(prepared):
                logits=output.logits[i,-1,job.candidate_ids].float().cpu().tolist()
                if not all(math.isfinite(v) for v in logits):
                    raise ValueError('nonfinite batched logits')
                result.append(JobScore(logits=logits,inputTokens=len(job.input_ids)))
        return result

    def score(self, engine, prepared):
        if not prepared:
            return []
        if len(prepared)==1 or self.enabled is False:
            return engine.score(prepared)
        if self.enabled is None:
            # First actual cutoff only: compare all symbols against proven singles.
            serial=engine.score(prepared)
            try:
                batched=self._batch(engine,prepared)
                self.enabled=equivalent(serial,batched)
                self.gate='passed' if self.enabled else 'parity_failed'
            except Exception:
                self.enabled=False
                self.gate='batch_failed'
                self._clear_cuda()
            # Preserve baseline outputs at validation cutoff regardless of gate.
            return serial
        try:
            return self._batch(engine,prepared)
        except Exception:
            self.enabled=False
            self.gate='later_batch_failed'
            self._clear_cuda()
            return engine.score(prepared)

    @staticmethod
    def _clear_cuda():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
