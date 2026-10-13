# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host input checks and deterministic routing fixtures for the composed MoE block."""

from collections.abc import Mapping

import torch


def assert_replicated_batch(x_mixed: torch.Tensor, num_tokens: int | None = None) -> None:
    """Require identical active rows in a ``[ranks, capacity, hidden]`` TP batch.

    Capacity rows after ``num_tokens`` are unused and may differ across ranks.
    Token order must match, as the shared expert sums same-row TP contributions.
    """
    if x_mixed.ndim != 3 or x_mixed.shape[0] < 1:
        raise ValueError("x_mixed must have shape [ranks, capacity, hidden] with at least one rank")
    capacity = x_mixed.shape[1]
    active = capacity if num_tokens is None else max(0, min(capacity, int(num_tokens)))
    for rank in range(1, x_mixed.shape[0]):
        if not torch.equal(x_mixed[rank, :active], x_mixed[0, :active]):
            raise AssertionError(
                f"x_mixed rank {rank} carries a different active token batch than rank 0: "
                "the shared expert requires identical TP activation rows"
            )


def validate_tp_inputs(tensors: Mapping[str, torch.Tensor | int]) -> None:
    """Check the composed layer's TP contract before dispatch or golden computation.

    Activations and RMSNorm weights must be replicated to form one shared expert.
    Router weights and correction biases are also replicated checkpoint tensors.
    Raw L2/L3 callers must establish this contract before invoking compiled code.
    """
    x_mixed = tensors["x_mixed"]
    assert_replicated_batch(x_mixed, tensors.get("num_tokens"))
    num_ranks, _, hidden = x_mixed.shape
    gate_weight = tensors["gate_weight"]
    if gate_weight.ndim != 3:
        raise ValueError("gate_weight must have shape [ranks, experts, hidden]")
    expected_shapes = {
        "norm_weight": (num_ranks, hidden),
        "gate_weight": (num_ranks, gate_weight.shape[1], hidden),
        "correction_bias": (num_ranks, gate_weight.shape[1]),
    }
    for name, shape in expected_shapes.items():
        value = tensors[name]
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")
        for rank in range(1, num_ranks):
            if not torch.equal(value[rank], value[0]):
                raise AssertionError(f"{name} rank {rank} differs from rank 0: TP requires replicated weights")


def make_correction_bias(num_ranks: int, num_experts: int, topk: int) -> torch.Tensor:
    """Replicate a global bias that fixes exactly ``topk`` winners across EP ranks."""
    if num_ranks < 1 or num_experts < 1 or num_experts % num_ranks:
        raise ValueError("num_experts must be positive and divisible by a positive num_ranks")
    if not 1 <= topk <= num_experts:
        raise ValueError("topk must be between 1 and num_experts")
    slots = torch.arange(topk)
    winners = (slots % num_ranks) * (num_experts // num_ranks) + slots // num_ranks
    bias = torch.full((num_experts,), -5.0, dtype=torch.float32)
    bias[winners] = 5.0
    return bias.unsqueeze(0).expand(num_ranks, num_experts).contiguous()
