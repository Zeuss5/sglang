from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import msgspec
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
    sample_dflash_proposal_from_logits,
    top_k_renorm_prob,
    top_p_renorm_prob,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.utils import is_cuda, is_musa

logger = logging.getLogger(__name__)

WEAVER_TREE_EXPAND_WIDTH = 8
WEAVER_TREE_BATCH_EXPAND_WIDTH = 8
WEAVER_TREE_BATCH_EXPAND_BUDGET_UNIT = 16


@triton.jit
def _weaver_candidate_frontier_kernel(
    logits_ptr,
    candidate_ids_ptr,
    prefix_score_ptr,
    node_depth_ptr,
    active_ptr,
    frontier_tokens_ptr,
    frontier_parents_ptr,
    frontier_depths_ptr,
    frontier_scores_ptr,
    frontier_logprobs_ptr,
    frontier_active_ptr,
    frontier_is_sampled_ptr,
    uniforms_ptr,
    node_pool_ids_ptr,
    node_pool_ms_ptr,
    slot_start,
    WIDTH: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    EXPAND_WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    FRONTIER_SLOTS: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    NUM_NODES: tl.constexpr,
    GREEDY: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_POOL)
    pool_mask = offsets < POOL_SIZE
    row_base = row * POOL_SIZE + offsets
    token_ids = tl.load(candidate_ids_ptr + row_base, mask=pool_mask, other=-1)
    scores = tl.load(logits_ptr + row_base, mask=pool_mask, other=-float("inf")).to(
        tl.float32
    )
    scores = tl.where((token_ids >= 0) & pool_mask, scores, -float("inf"))
    scores_pool = scores  # pristine pool logits; `scores` is consumed by the top-K loop
    parent_score = tl.load(prefix_score_ptr + row)
    parent_depth = tl.load(node_depth_ptr + row)
    parent_active = (tl.load(active_ptr + row) != 0) & (parent_depth < DEPTH)
    batch = row // WIDTH
    row_in_width = row - batch * WIDTH
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.where(scores == -float("inf"), 0.0, tl.exp(scores - max_score))
    log_denom = tl.log(tl.sum(exp_scores, axis=0)) + max_score
    child_base = batch * FRONTIER_SLOTS + (slot_start + row_in_width) * EXPAND_WIDTH
    child_depth = parent_depth + 1
    # UniVer node construction: (EXPAND_WIDTH - 1) deterministic top-K children plus
    # ONE child sampled from the residual draft distribution. The cascade identity
    # min(1, q/s) is only lossless for a candidate drawn from s, so a tree of pure
    # top-K cannot be verified losslessly at temperature > 0; the sampled child is
    # what restores it.
    best_norm = tl.full((), -float("inf"), dtype=tl.float32)
    for child in tl.static_range(0, EXPAND_WIDTH - 1):
        top_value, top_index = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        child_token = tl.load(candidate_ids_ptr + row * POOL_SIZE + top_index)
        child_valid = parent_active & (child_token >= 0) & (top_value != -float("inf"))
        norm_lp = top_value - log_denom
        if child == 0:
            best_norm = norm_lp
        out_index = child_base + child
        tl.store(frontier_tokens_ptr + out_index, tl.where(child_valid, child_token, 0))
        tl.store(
            frontier_parents_ptr + out_index,
            tl.where(child_valid, slot_start + row_in_width, 0),
        )
        tl.store(frontier_depths_ptr + out_index, tl.where(child_valid, child_depth, 0))
        tl.store(
            frontier_scores_ptr + out_index,
            tl.where(child_valid, parent_score + norm_lp, -float("inf")),
        )
        tl.store(
            frontier_logprobs_ptr + out_index,
            tl.where(child_valid, norm_lp, -float("inf")),
        )
        tl.store(frontier_active_ptr + out_index, child_valid)
        tl.store(frontier_is_sampled_ptr + out_index, False)
        scores = tl.where(offsets == top_index, -float("inf"), scores)

    if GREEDY:
        # Greedy verification does not need a sampled child: q is a point mass, so
        # s cancels out of the acceptance test and the tree is verified losslessly
        # either way. A residual draw would only spend a slot that the 8th top-K
        # token uses better -- measured 7-48% accepted length in simulation.
        top_value, top_index = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        child_token = tl.load(candidate_ids_ptr + row * POOL_SIZE + top_index)
        child_valid = parent_active & (child_token >= 0) & (top_value != -float("inf"))
        norm_lp = top_value - log_denom
        out_index = child_base + (EXPAND_WIDTH - 1)
        tl.store(frontier_tokens_ptr + out_index, tl.where(child_valid, child_token, 0))
        tl.store(
            frontier_parents_ptr + out_index,
            tl.where(child_valid, slot_start + row_in_width, 0),
        )
        tl.store(frontier_depths_ptr + out_index, tl.where(child_valid, child_depth, 0))
        tl.store(
            frontier_scores_ptr + out_index,
            tl.where(child_valid, parent_score + norm_lp, -float("inf")),
        )
        tl.store(
            frontier_logprobs_ptr + out_index,
            tl.where(child_valid, norm_lp, -float("inf")),
        )
        tl.store(frontier_active_ptr + out_index, child_valid)
        tl.store(frontier_is_sampled_ptr + out_index, False)
        return

    # `scores` now holds the pool with the deterministic children removed: exactly
    # the residual draft distribution ms. Sample one child from it by inverse CDF.
    resid_max = tl.max(scores, axis=0)
    resid_exp = tl.where(scores == -float("inf"), 0.0, tl.exp(scores - resid_max))
    resid_sum = tl.sum(resid_exp, axis=0)
    resid_ok = resid_sum > 0.0
    resid_probs = resid_exp / tl.maximum(resid_sum, 1e-20)
    cdf = tl.cumsum(resid_probs, axis=0)
    u = tl.load(uniforms_ptr + child_base)
    sel = tl.min(tl.where(cdf >= u, offsets, POOL_SIZE), axis=0)
    sel = tl.minimum(tl.maximum(sel, 0), POOL_SIZE - 1)
    samp_token = tl.load(candidate_ids_ptr + row * POOL_SIZE + sel)
    samp_p = tl.sum(tl.where(offsets == sel, resid_probs, 0.0), axis=0)
    samp_valid = parent_active & (samp_token >= 0) & resid_ok & (samp_p > 0.0)
    out_index = child_base + (EXPAND_WIDTH - 1)
    tl.store(frontier_tokens_ptr + out_index, tl.where(samp_valid, samp_token, 0))
    tl.store(
        frontier_parents_ptr + out_index,
        tl.where(samp_valid, slot_start + row_in_width, 0),
    )
    tl.store(frontier_depths_ptr + out_index, tl.where(samp_valid, child_depth, 0))
    # Scored on its OWN pool-relative probability, like any sibling. Scoring it at
    # the best sibling's logprob (as this once did) does not merely protect it from
    # pruning -- frontier_scores also decides which node is expanded next, so a
    # random residual draw was competing for expansion as the node's best child and
    # the tree grew along random tokens. Losing the sampled child is handled in
    # verification instead: with no sampled child the node uses Z_v = 1, which makes
    # UniVer degenerate to membership testing -- lossless, and optimal for a
    # deterministic candidate set.
    samp_pool_lp = tl.sum(tl.where(offsets == sel, scores_pool, 0.0), axis=0) - log_denom
    tl.store(
        frontier_scores_ptr + out_index,
        tl.where(samp_valid, parent_score + samp_pool_lp, -float("inf")),
    )
    # Stored relative to the RESIDUAL, i.e. this is log ms(u_m) — the denominator
    # UniVer's acceptance term needs, not the full-pool logprob.
    tl.store(
        frontier_logprobs_ptr + out_index,
        tl.where(samp_valid, tl.log(tl.maximum(samp_p, 1e-20)), -float("inf")),
    )
    tl.store(frontier_active_ptr + out_index, samp_valid)
    tl.store(frontier_is_sampled_ptr + out_index, samp_valid)

    # Persist this node's pool and residual ms. Verification needs
    #   Z_v = 1 - p~ + sum_{x in pool}[p~q(x) - ms(x)]_+ + p~*(1 - Q_pool)
    # which couples build-time ms against target probs that do not exist yet, so
    # neither side can precompute it and ms has to survive to the verify kernel.
    node_slot = slot_start + row_in_width
    pool_base = (batch * NUM_NODES + node_slot) * POOL_SIZE
    tl.store(node_pool_ids_ptr + pool_base + offsets,
             tl.where(pool_mask, token_ids, 0), mask=pool_mask)
    tl.store(node_pool_ms_ptr + pool_base + offsets,
             tl.where(pool_mask & parent_active, resid_probs, 0.0), mask=pool_mask)

@triton.jit
def _weaver_indexed_attention_kernel(
    q_ptr,
    current_keys_ptr,
    current_values_ptr,
    external_keys_ptr,
    external_values_ptr,
    external_mask_ptr,
    node_keys_ptr,
    node_values_ptr,
    parent_ancestors_ptr,
    row_batch_indices_ptr,
    position_ids_ptr,
    out_ptr,
    LAYER: tl.constexpr,
    PREFIX: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_NODES: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_CTX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    ctx = tl.arange(0, BLOCK_CTX)
    ctx_offsets = ctx[:, None]
    dim_offsets = tl.arange(0, BLOCK_D)[None, :]
    dim_mask = dim_offsets < HEAD_DIM
    batch = tl.load(row_batch_indices_ptr + row)
    pos = tl.load(position_ids_ptr + row)
    pos = tl.minimum(tl.maximum(pos, 0), DEPTH - 1)

    q = tl.load(
        q_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    ext_pos = ctx_offsets
    tok_pos = ctx_offsets - PREFIX
    tok_pos_1d = ctx - PREFIX
    ext_pos_safe = tl.minimum(ctx_offsets, PREFIX - 1)
    ext_pos_1d_safe = tl.minimum(ctx, PREFIX - 1)
    tok_pos_1d_safe = tl.maximum(tok_pos_1d, 0)
    ext_ctx = ctx_offsets < PREFIX
    tok_ctx_1d = (ctx >= PREFIX) & (tok_pos_1d <= pos)
    tok_current_1d = tok_ctx_1d & (tok_pos_1d == pos)
    tok_ancestor_1d = tok_ctx_1d & (tok_pos_1d < pos)
    ancestor_slot = tl.load(
        parent_ancestors_ptr + row * DEPTH + tok_pos_1d_safe,
        mask=tok_ancestor_1d,
        other=-1,
    )
    ancestor_valid_1d = tok_ancestor_1d & (ancestor_slot >= 0)
    ancestor_slot = tl.maximum(ancestor_slot, 0)
    tok_current = tok_current_1d[:, None]
    ancestor_valid = ancestor_valid_1d[:, None]
    ancestor_slot = ancestor_slot[:, None]

    ext_key = tl.load(
        external_keys_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_key = tl.load(
        current_keys_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    ancestor_key = tl.load(
        node_keys_ptr
        + (
            (
                ((batch * NUM_NODES + ancestor_slot) * NUM_LAYERS + LAYER)
                * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ancestor_valid & dim_mask,
        other=0.0,
    )
    tok_key = tl.where(tok_current, current_key, ancestor_key)
    key = tl.where(ext_ctx, ext_key, tok_key).to(tl.float32)
    scores = tl.dot(key.to(tl.bfloat16), tl.trans(q.to(tl.bfloat16)))
    scores = (
        scores.to(tl.bfloat16) * tl.rsqrt(HEAD_DIM + 0.0)
    ).to(tl.bfloat16).to(tl.float32)
    ext_mask = (
        tl.load(
        external_mask_ptr + batch * PREFIX + ext_pos_1d_safe,
        mask=ctx < PREFIX,
        other=0,
        )
        != 0
    )
    ctx_valid = (
        ((ctx < PREFIX) & ext_mask)
        | ((ctx >= PREFIX) & ((ctx - PREFIX) == pos))
        | ancestor_valid_1d
    )
    scores = tl.where(ctx_valid[:, None], scores, -float("inf"))
    max_score = tl.max(scores, axis=0)
    probs = tl.exp(scores - max_score)
    probs = probs / tl.sum(probs, axis=0)
    probs = probs.to(tl.bfloat16)

    ext_value = tl.load(
        external_values_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_value = tl.load(
        current_values_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    ancestor_value = tl.load(
        node_values_ptr
        + (
            (
                ((batch * NUM_NODES + ancestor_slot) * NUM_LAYERS + LAYER)
                * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ancestor_valid & dim_mask,
        other=0.0,
    )
    tok_value = tl.where(tok_current, current_value, ancestor_value)
    ext_probs = tl.where(ext_ctx, probs, 0.0).to(tl.bfloat16)
    tok_probs = tl.where(ctx_offsets >= PREFIX, probs, 0.0).to(tl.bfloat16)
    ext_out = tl.dot(tl.trans(ext_probs), ext_value).to(tl.bfloat16)
    tok_out = tl.dot(tl.trans(tok_probs), tok_value).to(tl.bfloat16)
    out = (ext_out + tok_out).to(tl.bfloat16)
    tl.store(
        out_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        out,
        mask=dim_mask,
    )


@triton.jit
def _weaver_chain_attention_kernel(
    q_ptr,
    current_keys_ptr,
    current_values_ptr,
    external_keys_ptr,
    external_values_ptr,
    external_mask_ptr,
    chain_keys_ptr,
    chain_values_ptr,
    position_ids_ptr,
    out_ptr,
    LAYER: tl.constexpr,
    PREFIX: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_CTX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    ctx = tl.arange(0, BLOCK_CTX)
    ctx_offsets = ctx[:, None]
    dim_offsets = tl.arange(0, BLOCK_D)[None, :]
    dim_mask = dim_offsets < HEAD_DIM
    pos = tl.load(position_ids_ptr + batch)
    pos = tl.minimum(tl.maximum(pos, 0), DEPTH - 1)

    q = tl.load(
        q_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    ext_pos = ctx_offsets
    tok_pos = ctx_offsets - PREFIX
    tok_pos_1d = ctx - PREFIX
    ext_pos_safe = tl.minimum(ctx_offsets, PREFIX - 1)
    ext_pos_1d_safe = tl.minimum(ctx, PREFIX - 1)
    tok_pos_safe = tl.maximum(tl.minimum(tok_pos, DEPTH - 1), 0)
    tok_ctx_1d = (ctx >= PREFIX) & (tok_pos_1d <= pos)
    tok_current = (tok_ctx_1d & (tok_pos_1d == pos))[:, None]
    tok_history = (tok_ctx_1d & (tok_pos_1d < pos))[:, None]
    ext_ctx = ctx_offsets < PREFIX

    ext_key = tl.load(
        external_keys_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_key = tl.load(
        current_keys_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    history_key = tl.load(
        chain_keys_ptr
        + (
            (
                ((batch * DEPTH + tok_pos_safe) * NUM_LAYERS + LAYER) * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=tok_history & dim_mask,
        other=0.0,
    )
    tok_key = tl.where(tok_current, current_key, history_key)
    key = tl.where(ext_ctx, ext_key, tok_key).to(tl.float32)
    scores = tl.dot(key.to(tl.bfloat16), tl.trans(q.to(tl.bfloat16)))
    scores = (
        scores.to(tl.bfloat16) * tl.rsqrt(HEAD_DIM + 0.0)
    ).to(tl.bfloat16).to(tl.float32)
    ext_mask = (
        tl.load(
            external_mask_ptr + batch * PREFIX + ext_pos_1d_safe,
            mask=ctx < PREFIX,
            other=0,
        )
        != 0
    )
    ctx_valid = ((ctx < PREFIX) & ext_mask) | tok_ctx_1d
    scores = tl.where(ctx_valid[:, None], scores, -float("inf"))
    max_score = tl.max(scores, axis=0)
    probs = tl.exp(scores - max_score)
    probs = probs / tl.sum(probs, axis=0)
    probs = probs.to(tl.bfloat16)

    ext_value = tl.load(
        external_values_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_value = tl.load(
        current_values_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    history_value = tl.load(
        chain_values_ptr
        + (
            (
                ((batch * DEPTH + tok_pos_safe) * NUM_LAYERS + LAYER) * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=tok_history & dim_mask,
        other=0.0,
    )
    tok_value = tl.where(tok_current, current_value, history_value)
    ext_probs = tl.where(ext_ctx, probs, 0.0).to(tl.bfloat16)
    tok_probs = tl.where(ctx_offsets >= PREFIX, probs, 0.0).to(tl.bfloat16)
    ext_out = tl.dot(tl.trans(ext_probs), ext_value).to(tl.bfloat16)
    tok_out = tl.dot(tl.trans(tok_probs), tok_value).to(tl.bfloat16)
    out = (ext_out + tok_out).to(tl.bfloat16)
    tl.store(
        out_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        out,
        mask=dim_mask,
    )


@triton.jit
def _tree_metadata_parent_chain_kernel(
    parent_indices_ptr,
    node_mask_ptr,
    prefix_lens_ptr,
    mask_offsets_ptr,
    custom_mask_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    NUM_NODES: tl.constexpr,
    CHAIN_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < NUM_NODES
    row_base = batch * NUM_NODES
    row_valid = tl.load(node_mask_ptr + row_base + row) != 0
    prefix_len = tl.load(prefix_lens_ptr + batch)
    width = prefix_len + NUM_NODES

    ancestor = tl.full((BLOCK_N,), False, dtype=tl.int1)
    current = row
    current_valid = row_valid
    for _ in tl.static_range(0, CHAIN_STEPS):
        ancestor |= current_valid & col_mask & (cols == current)
        parent = tl.load(
            parent_indices_ptr + row_base + current,
            mask=current_valid,
            other=-1,
        )
        current_valid = current_valid & (parent >= 0) & (parent < NUM_NODES)
        current = tl.maximum(tl.minimum(parent, NUM_NODES - 1), 0)

    mask_offset = tl.load(mask_offsets_ptr + batch) + row * width + prefix_len + cols
    tl.store(custom_mask_ptr + mask_offset, ancestor, mask=col_mask)

    upper = col_mask & (cols > row)
    rows = tl.full((BLOCK_N,), row, dtype=tl.int64)
    descendant = tl.full((BLOCK_N,), False, dtype=tl.int1)
    current = cols
    col_valid = tl.load(node_mask_ptr + row_base + cols, mask=col_mask, other=0) != 0
    current_valid = upper & col_valid & row_valid
    for _ in tl.static_range(0, CHAIN_STEPS):
        descendant |= current_valid & (current == rows)
        parent = tl.load(
            parent_indices_ptr + row_base + current,
            mask=current_valid,
            other=-1,
        )
        current_valid = current_valid & (parent >= 0) & (parent < NUM_NODES)
        current = tl.maximum(tl.minimum(parent, NUM_NODES - 1), 0)
    next_token_values = tl.where(descendant, cols, NUM_NODES)
    next_token = tl.min(next_token_values, axis=0)
    next_token = tl.where(next_token == NUM_NODES, -1, next_token)

    row_parent = tl.load(parent_indices_ptr + row_base + row)
    col_parent = tl.load(parent_indices_ptr + row_base + cols, mask=col_mask, other=-2)
    sibling = (
        upper
        & row_valid
        & (row_parent >= 0)
        & (col_parent == row_parent)
        & col_valid
    )
    next_sibling_values = tl.where(sibling, cols, NUM_NODES)
    next_sibling = tl.min(next_sibling_values, axis=0)
    next_sibling = tl.where(next_sibling == NUM_NODES, -1, next_sibling)

    tl.store(retrieve_next_token_ptr + row_base + row, next_token)
    tl.store(retrieve_next_sibling_ptr + row_base + row, next_sibling)


@triton.jit
def _weaver_traversal_verify_kernel(
    candidates_ptr,
    parent_indices_ptr,
    depths_ptr,
    node_mask_ptr,
    draft_logprobs_ptr,
    target_probs_ptr,
    uniform_samples_ptr,
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    accept_leaf_ptr,
    NUM_NODES: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    col_mask = offsets < NUM_NODES
    row_base = batch * NUM_NODES

    tl.store(
        predicts_ptr + row_base + offsets,
        tl.full((BLOCK_N,), -1, dtype=tl.int32),
        mask=col_mask,
    )
    tl.store(
        accept_index_ptr + row_base + offsets,
        tl.full((BLOCK_N,), -1, dtype=tl.int32),
        mask=col_mask,
    )

    parents = tl.load(parent_indices_ptr + row_base + offsets, mask=col_mask, other=-1)
    depths = tl.load(depths_ptr + row_base + offsets, mask=col_mask, other=0)
    tokens = tl.load(candidates_ptr + row_base + offsets, mask=col_mask, other=0)
    active = (tl.load(node_mask_ptr + row_base + offsets, mask=col_mask, other=0) != 0) & col_mask
    active = active | (offsets == 0)

    local_logprobs = tl.load(
        draft_logprobs_ptr + row_base + offsets,
        mask=col_mask,
        other=-float("inf"),
    ).to(tl.float32)
    local_weights = tl.where((offsets > 0) & active, tl.exp(local_logprobs), 0.0)
    draft_probs = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for node in tl.range(1, NUM_NODES, loop_unroll_factor=1):
        node_parent = tl.load(parent_indices_ptr + row_base + node)
        node_weight = tl.load(draft_logprobs_ptr + row_base + node).to(tl.float32)
        node_weight = tl.exp(node_weight)
        sibling_weight = tl.sum(
            tl.where((parents == node_parent) & active & (offsets > 0), local_weights, 0.0),
            axis=0,
        )
        node_prob = node_weight / tl.maximum(sibling_weight, 1.0e-20)
        node_is_active = (tl.load(node_mask_ptr + row_base + node) != 0) & (sibling_weight > 0.0)
        draft_probs = tl.where((offsets == node) & node_is_active, node_prob, draft_probs)

    node_p = tl.where(offsets == 0, 1.0, 0.0).to(tl.float32)
    node_p_valid = offsets == 0
    accept_leaf = tl.full((), 0, dtype=tl.int64)
    done = tl.full((), False, dtype=tl.int1)

    verify_step = tl.full((), 0, dtype=tl.int64)
    while (verify_step < NUM_NODES) & (~done):
        cur = tl.full((), 0, dtype=tl.int64)
        cur_p = tl.full((), 1.0, dtype=tl.float32)
        parent_for_leaf = tl.full((), 0, dtype=tl.int64)
        p_parent_for_leaf = tl.full((), 1.0, dtype=tl.float32)
        leaf = tl.full((), 0, dtype=tl.int64)
        leaf_p = tl.full((), 1.0, dtype=tl.float32)
        descending = ~done

        descend_step = tl.full((), 0, dtype=tl.int64)
        while (descend_step < NUM_NODES) & descending:
            child_values = tl.where(active & (parents == cur), offsets, NUM_NODES)
            child = tl.min(child_values, axis=0)
            has_child = child < NUM_NODES
            take_child = descending & has_child
            take_leaf = descending & (~has_child)

            child_safe = tl.minimum(tl.maximum(child, 0), NUM_NODES - 1)
            child_token = tl.load(candidates_ptr + row_base + child_safe)
            child_token_safe = tl.minimum(tl.maximum(child_token, 0), VOCAB_SIZE - 1)
            child_q = tl.load(
                target_probs_ptr + (row_base + cur) * VOCAB_SIZE + child_token_safe,
                mask=take_child,
                other=0.0,
            ).to(tl.float32)
            child_s = tl.sum(tl.where(offsets == child, draft_probs, 0.0), axis=0)
            computed_child_p = tl.minimum(
                cur_p * child_q / tl.maximum(child_s, 1.0e-20),
                1.0,
            )
            stored_child_p = tl.sum(tl.where(offsets == child, node_p, 0.0), axis=0)
            stored_child_valid = tl.sum(
                tl.where(offsets == child, node_p_valid.to(tl.int32), 0),
                axis=0,
            ) != 0
            next_child_p = tl.where(stored_child_valid, stored_child_p, computed_child_p)

            leaf = tl.where(take_leaf, cur, leaf)
            leaf_p = tl.where(take_leaf, cur_p, leaf_p)
            parent_for_leaf = tl.where(take_child, cur, parent_for_leaf)
            p_parent_for_leaf = tl.where(take_child, cur_p, p_parent_for_leaf)
            cur = tl.where(take_child, child, cur)
            cur_p = tl.where(take_child, next_child_p, cur_p)
            descending = descending & has_child
            descend_step += 1

        eta = tl.load(uniform_samples_ptr + row_base + verify_step, mask=~done, other=0.0)
        accept_now = (~done) & ((leaf == 0) | (eta < leaf_p))
        reject_now = (~done) & (~accept_now)
        accept_leaf = tl.where(accept_now, leaf, accept_leaf)

        leaf_safe = tl.minimum(tl.maximum(leaf, 0), NUM_NODES - 1)
        reject_parent = tl.load(parent_indices_ptr + row_base + leaf_safe, mask=reject_now, other=0)
        reject_parent = tl.minimum(tl.maximum(reject_parent, 0), NUM_NODES - 1)

        child_mask = active & (parents == reject_parent)
        child_tokens = tl.minimum(tl.maximum(tokens, 0), VOCAB_SIZE - 1)
        q_children = tl.load(
            target_probs_ptr + (row_base + reject_parent) * VOCAB_SIZE + child_tokens,
            mask=child_mask & reject_now,
            other=0.0,
        ).to(tl.float32)
        q_sum = tl.sum(tl.where(child_mask, q_children, 0.0), axis=0)
        positive = tl.maximum(p_parent_for_leaf * q_children - draft_probs, 0.0)
        positive_sum = tl.sum(tl.where(child_mask, positive, 0.0), axis=0)
        target_tail = tl.maximum(p_parent_for_leaf * (1.0 - q_sum), 0.0)
        residual_mass = positive_sum + target_tail
        new_parent_p = residual_mass / tl.maximum(
            residual_mass + 1.0 - p_parent_for_leaf,
            1.0e-20,
        )

        rejected_s = tl.sum(tl.where(offsets == leaf, draft_probs, 0.0), axis=0)
        renorm = 1.0 / tl.maximum(1.0 - rejected_s, 1.0e-20)
        draft_probs = tl.where(
            reject_now & child_mask & (offsets != leaf),
            draft_probs * renorm,
            draft_probs,
        )
        draft_probs = tl.where(reject_now & (offsets == leaf), 0.0, draft_probs)
        active = tl.where(reject_now & (offsets == leaf), False, active)
        node_p = tl.where(reject_now & (offsets == reject_parent), new_parent_p, node_p)
        node_p_valid = node_p_valid | (reject_now & (offsets == reject_parent))
        done = done | accept_now
        verify_step += 1

    accept_leaf = tl.minimum(tl.maximum(accept_leaf, 0), NUM_NODES - 1)
    tl.store(accept_leaf_ptr + batch, accept_leaf)
    leaf_depth = tl.load(depths_ptr + row_base + accept_leaf).to(tl.int32)
    tl.store(accept_token_num_ptr + batch, leaf_depth)

    chain_node = accept_leaf
    chain_step = tl.full((), 0, dtype=tl.int64)
    while (chain_step < NUM_NODES) & (chain_node >= 0):
        chain_valid = chain_node >= 0
        chain_safe = tl.minimum(tl.maximum(chain_node, 0), NUM_NODES - 1)
        chain_depth = tl.load(depths_ptr + row_base + chain_safe, mask=chain_valid, other=0)
        tl.store(
            accept_index_ptr + row_base + chain_depth,
            (row_base + chain_safe).to(tl.int32),
            mask=chain_valid & (chain_depth < NUM_NODES),
        )
        parent = tl.load(parent_indices_ptr + row_base + chain_safe, mask=chain_valid, other=-1)
        parent_safe = tl.minimum(tl.maximum(parent, 0), NUM_NODES - 1)
        token = tl.load(candidates_ptr + row_base + chain_safe, mask=chain_valid, other=0)
        tl.store(
            predicts_ptr + row_base + parent_safe,
            token.to(tl.int32),
            mask=chain_valid & (parent >= 0),
        )
        chain_node = tl.where(chain_valid, parent, chain_node)
        chain_step += 1


@triton.jit
def _univer_verify_kernel(
    candidates_ptr,
    parent_indices_ptr,
    depths_ptr,
    node_mask_ptr,
    draft_logprobs_ptr,
    is_sampled_ptr,
    pool_ids_ptr,
    pool_ms_ptr,
    target_probs_ptr,
    uniform_samples_ptr,
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    accept_leaf_ptr,
    residual_ptr,
    residual_valid_ptr,
    NUM_NODES: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """UniVer verification (arXiv 2605.04543).

    The cascade this replaces accepts with min(1, cur_p*q/s), an identity that is
    lossless only for a candidate drawn with probability s. Our children are the
    drafter's top-K plus one residual sample, so it needs a rule built for a mixed
    deterministic/sampled candidate set.

    Allocation runs top-down (parent[i] < i, so index order is topological):
        Z_v    = 1 - p~ + sum_{x in pool}[p~q(x) - ms(x)]_+ + p~*(1 - Q_pool)
        p(u_m) = min(1, p~*q(u_m)/ms(u_m))
        p(u_k) = [p~*q(u_k) - ms(u_k)]_+ * (1 - p(u_m)) / Z_v   -> ms(u_k) = 0
        p(!v)  = (1 - p(u_m))(1 - p~) / Z_v
    then conditional normalisation over children in index order with the sampled
    child last. Decision is a post-order walk: a leaf fires on eta < p~_v, an
    interior node on eta < p~_v^res.
    """
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    col_mask = offsets < NUM_NODES
    row_base = batch * NUM_NODES
    pool_off = tl.arange(0, BLOCK_POOL)
    pool_mask = pool_off < POOL_SIZE

    tl.store(
        predicts_ptr + row_base + offsets,
        tl.full((BLOCK_N,), -1, dtype=tl.int32),
        mask=col_mask,
    )
    tl.store(
        accept_index_ptr + row_base + offsets,
        tl.full((BLOCK_N,), -1, dtype=tl.int32),
        mask=col_mask,
    )

    parents = tl.load(parent_indices_ptr + row_base + offsets, mask=col_mask, other=-1)
    tokens = tl.load(candidates_ptr + row_base + offsets, mask=col_mask, other=0)
    active = (
        tl.load(node_mask_ptr + row_base + offsets, mask=col_mask, other=0) != 0
    ) & col_mask
    active = active | (offsets == 0)
    sampled = (
        tl.load(is_sampled_ptr + row_base + offsets, mask=col_mask, other=0) != 0
    ) & col_mask
    lps = tl.load(
        draft_logprobs_ptr + row_base + offsets, mask=col_mask, other=-float("inf")
    ).to(tl.float32)
    tok_safe = tl.minimum(tl.maximum(tokens, 0), VOCAB_SIZE - 1)

    p_tilde = tl.where(offsets == 0, 1.0, 0.0).to(tl.float32)
    p_res = tl.zeros((BLOCK_N,), dtype=tl.float32)
    has_kids = offsets < 0

    for v in tl.range(0, NUM_NODES, loop_unroll_factor=1):
        pt = tl.sum(tl.where(offsets == v, p_tilde, 0.0), axis=0)
        child_mask = active & (parents == v) & (offsets > 0)
        n_kids = tl.sum(tl.where(child_mask, 1, 0), axis=0)

        pb = (row_base + v) * POOL_SIZE
        ids = tl.load(pool_ids_ptr + pb + pool_off, mask=pool_mask, other=0)
        ids = tl.minimum(tl.maximum(ids, 0), VOCAB_SIZE - 1)
        ms = tl.load(pool_ms_ptr + pb + pool_off, mask=pool_mask, other=0.0).to(
            tl.float32
        )
        qp = tl.load(
            target_probs_ptr + (row_base + v) * VOCAB_SIZE + ids,
            mask=pool_mask,
            other=0.0,
        ).to(tl.float32)
        # Off-pool target mass rides in as p~*(1 - Q_pool): ms is zero there, so
        # every such x contributes p~*q(x) to Z_v.
        q_pool = tl.sum(tl.where(pool_mask, qp, 0.0), axis=0)
        s_pos = tl.sum(
            tl.where(pool_mask, tl.maximum(pt * qp - ms, 0.0), 0.0), axis=0
        )
        z_v = tl.maximum(1.0 - pt + s_pos + pt * (1.0 - q_pool), 1.0e-20)

        um = tl.min(tl.where(child_mask & sampled, offsets, NUM_NODES), axis=0)
        has_um = um < NUM_NODES
        um_safe = tl.minimum(tl.maximum(um, 0), NUM_NODES - 1)
        ms_um = tl.exp(tl.sum(tl.where(offsets == um_safe, lps, 0.0), axis=0))
        tok_um = tl.sum(tl.where(offsets == um_safe, tok_safe, 0), axis=0)
        q_um = tl.load(
            target_probs_ptr + (row_base + v) * VOCAB_SIZE + tok_um
        ).to(tl.float32)
        p_um = tl.where(
            has_um & (ms_um > 0.0),
            tl.minimum(pt * q_um / tl.maximum(ms_um, 1.0e-20), 1.0),
            0.0,
        )
        # Without a sampled child the node's candidates are a fixed set, and the
        # optimal lossless rule is membership: emit y ~ q and descend if y is a
        # child. That is exactly this allocation with Z_v = 1, giving
        # p(u_k) = p~*q(u_k) and a residual of p~*q elsewhere.
        z_v = tl.where(has_um, z_v, 1.0)

        det_mask = child_mask & (~sampled)
        q_det = tl.load(
            target_probs_ptr + (row_base + v) * VOCAB_SIZE + tok_safe,
            mask=det_mask,
            other=0.0,
        ).to(tl.float32)
        p_det = tl.where(det_mask, pt * q_det * (1.0 - p_um) / z_v, 0.0)

        # The conditional scaling p(u_j)/(1 - sum_{i<j} p(u_i)) must run in the
        # order the decision phase TESTS the children, which is node-index order.
        # The sampled child is scored above its siblings so it is selected first
        # and lands at the LOWEST index -- it is not last. Take the prefix over all
        # children in index order, wherever the sampled one happens to sit.
        p_all = tl.where(
            det_mask, p_det, tl.where((offsets == um_safe) & has_um, p_um, 0.0)
        )
        incl = tl.cumsum(p_all, axis=0)
        prev = incl - p_all
        pt_all = tl.where(
            child_mask, p_all / tl.maximum(1.0 - prev, 1.0e-20), 0.0
        )
        p_tilde = tl.where(
            child_mask, tl.minimum(tl.maximum(pt_all, 0.0), 1.0), p_tilde
        )

        p_not = (1.0 - p_um) * (1.0 - pt) / z_v
        total = tl.sum(p_all, axis=0)
        pres_v = tl.minimum(
            tl.maximum(1.0 - p_not / tl.maximum(1.0 - total, 1.0e-20), 0.0), 1.0
        )
        p_res = tl.where(offsets == v, pres_v, p_res)
        has_kids = has_kids | ((offsets == v) & (n_kids > 0))

    accept = tl.full((), 0, dtype=tl.int64)
    got = tl.full((), False, dtype=tl.int1)
    cur = tl.full((), 0, dtype=tl.int64)
    descending = tl.full((), True, dtype=tl.int1)
    stopped = tl.full((), False, dtype=tl.int1)

    for _ in tl.range(0, 3 * NUM_NODES, loop_unroll_factor=1):
        cur_safe = tl.minimum(tl.maximum(cur, 0), NUM_NODES - 1)
        fc = tl.min(
            tl.where(active & (parents == cur_safe) & (offsets > 0), offsets, NUM_NODES),
            axis=0,
        )
        live = (~got) & (~stopped)
        go_down = live & descending & (fc < NUM_NODES)
        test_now = live & (~descending)

        nonleaf = (
            tl.sum(tl.where(offsets == cur_safe, has_kids.to(tl.int32), 0), axis=0) != 0
        )
        prob = tl.where(
            nonleaf,
            tl.sum(tl.where(offsets == cur_safe, p_res, 0.0), axis=0),
            tl.sum(tl.where(offsets == cur_safe, p_tilde, 0.0), axis=0),
        )
        eta = tl.load(uniform_samples_ptr + row_base + cur_safe)
        fires = test_now & (eta < prob)
        accept = tl.where(fires, cur, accept)
        got = got | fires

        # Fill with 0, not -1: a -1 filler sums across the whole block and turns
        # this into parents[cur] - (BLOCK_N - 1).
        par = tl.sum(tl.where(offsets == cur_safe, parents, 0), axis=0)
        ns = tl.min(
            tl.where(active & (parents == par) & (offsets > cur_safe), offsets, NUM_NODES),
            axis=0,
        )
        has_ns = (ns < NUM_NODES) & (par >= 0)
        move_sib = test_now & (~fires) & has_ns
        move_up = test_now & (~fires) & (~has_ns)

        cur = tl.where(go_down, fc, tl.where(move_sib, ns, tl.where(move_up, par, cur)))
        descending = tl.where(
            go_down | move_sib,
            True,
            tl.where(move_up | (descending & (fc >= NUM_NODES)), False, descending),
        )
        stopped = stopped | (move_up & (par < 0))

    accept = tl.minimum(tl.maximum(accept, 0), NUM_NODES - 1)
    tl.store(accept_leaf_ptr + batch, accept)
    leaf_depth = tl.load(depths_ptr + row_base + accept).to(tl.int32)
    tl.store(accept_token_num_ptr + batch, leaf_depth)

    # An interior node fires from its residual, not from q(.|v). Materialise that
    # distribution so the bonus-token sampler draws the right thing.
    acc_nonleaf = (
        tl.sum(tl.where(offsets == accept, has_kids.to(tl.int32), 0), axis=0) != 0
    )
    tl.store(residual_valid_ptr + batch, acc_nonleaf.to(tl.int32))
    pt_acc = tl.sum(tl.where(offsets == accept, p_tilde, 0.0), axis=0)
    for start in tl.range(0, VOCAB_SIZE, BLOCK_V, loop_unroll_factor=1):
        voff = start + tl.arange(0, BLOCK_V)
        vmask = voff < VOCAB_SIZE
        qv = tl.load(
            target_probs_ptr + (row_base + accept) * VOCAB_SIZE + voff,
            mask=vmask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            residual_ptr + batch * VOCAB_SIZE + voff,
            tl.where(acc_nonleaf, pt_acc * qv, 0.0),
            mask=vmask,
        )
    pb = (row_base + accept) * POOL_SIZE
    ids = tl.load(pool_ids_ptr + pb + pool_off, mask=pool_mask, other=0)
    ids = tl.minimum(tl.maximum(ids, 0), VOCAB_SIZE - 1)
    ms = tl.load(pool_ms_ptr + pb + pool_off, mask=pool_mask, other=0.0).to(tl.float32)
    cur_r = tl.load(
        residual_ptr + batch * VOCAB_SIZE + ids, mask=pool_mask & acc_nonleaf, other=0.0
    )
    tl.store(
        residual_ptr + batch * VOCAB_SIZE + ids,
        tl.maximum(cur_r - ms, 0.0),
        mask=pool_mask & acc_nonleaf,
    )
    # Only the deterministic children still PRESENT are covered by their own
    # allocation. Pruned ones keep their mass here, or it vanishes and the output
    # distribution drifts.
    present_det = active & (parents == accept) & (offsets > 0) & (~sampled)
    tl.store(
        residual_ptr + batch * VOCAB_SIZE + tok_safe,
        tl.zeros((BLOCK_N,), dtype=tl.float32),
        mask=present_det & acc_nonleaf,
    )

    chain_node = accept
    for _ in tl.range(0, NUM_NODES, loop_unroll_factor=1):
        chain_valid = chain_node >= 0
        chain_safe = tl.minimum(tl.maximum(chain_node, 0), NUM_NODES - 1)
        chain_depth = tl.load(
            depths_ptr + row_base + chain_safe, mask=chain_valid, other=0
        )
        tl.store(
            accept_index_ptr + row_base + chain_depth,
            (row_base + chain_safe).to(tl.int32),
            mask=chain_valid & (chain_depth < NUM_NODES),
        )
        parent = tl.load(
            parent_indices_ptr + row_base + chain_safe, mask=chain_valid, other=-1
        )
        parent_safe = tl.minimum(tl.maximum(parent, 0), NUM_NODES - 1)
        token = tl.load(
            candidates_ptr + row_base + chain_safe, mask=chain_valid, other=0
        )
        tl.store(
            predicts_ptr + row_base + parent_safe,
            token.to(tl.int32),
            mask=chain_valid & (parent >= 0),
        )
        chain_node = tl.where(chain_valid, parent, chain_node)


@triton.jit
def _weaver_current_cache_write_kernel(
    current_keys_ptr,
    current_values_ptr,
    node_keys_ptr,
    node_values_ptr,
    parent_ancestors_ptr,
    slot_ancestors_ptr,
    valid_ptr,
    node_depth_ptr,
    slot_start,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_NODES: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOTAL_KV: tl.constexpr,
    TOTAL_ANCESTORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kv_mask = offsets < TOTAL_KV
    hd = offsets % HEAD_DIM
    head = (offsets // HEAD_DIM) % NUM_HEADS
    layer = (offsets // (HEAD_DIM * NUM_HEADS)) % NUM_LAYERS
    row_in_width = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS)) % WIDTH
    batch = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS * WIDTH)) % BS
    row = batch * WIDTH + row_in_width
    valid = tl.load(valid_ptr + row, mask=kv_mask, other=0) != 0
    current_index = (((layer * BS * WIDTH + row) * NUM_HEADS + head) * HEAD_DIM + hd)
    slot = slot_start + row_in_width
    node_index = (
        ((((batch * NUM_NODES + slot) * NUM_LAYERS + layer) * NUM_HEADS + head)
        * HEAD_DIM
        + hd)
    )
    key_value = tl.load(current_keys_ptr + current_index, mask=kv_mask & valid, other=0.0)
    value_value = tl.load(current_values_ptr + current_index, mask=kv_mask & valid, other=0.0)
    tl.store(node_keys_ptr + node_index, key_value, mask=kv_mask)
    tl.store(node_values_ptr + node_index, value_value, mask=kv_mask)

    ancestor_mask = offsets < TOTAL_ANCESTORS
    ancestor_depth = offsets % DEPTH
    ancestor_row = (offsets // DEPTH) % WIDTH
    ancestor_batch = (offsets // (DEPTH * WIDTH)) % BS
    ancestor_flat_row = ancestor_batch * WIDTH + ancestor_row
    ancestor_valid = (
        tl.load(
        valid_ptr + ancestor_flat_row,
        mask=ancestor_mask,
        other=0,
        )
        != 0
    )
    current_pos = tl.load(
        node_depth_ptr + ancestor_flat_row,
        mask=ancestor_mask,
        other=0,
    )
    current_pos = tl.minimum(current_pos, DEPTH - 1)
    parent_value = tl.load(parent_ancestors_ptr + offsets, mask=ancestor_mask, other=-1)
    ancestor_slot = slot_start + ancestor_row
    ancestor_value = tl.where(ancestor_depth == current_pos, ancestor_slot, parent_value)
    ancestor_value = tl.where(ancestor_valid, ancestor_value, -1)
    out_index = (ancestor_batch * NUM_NODES + ancestor_slot) * DEPTH + ancestor_depth
    tl.store(slot_ancestors_ptr + out_index, ancestor_value, mask=ancestor_mask)


def weaver_tree_expand_width() -> int:
    """Children generated per expanded node.

    Hardcoded to 8 upstream. DARTree (arXiv 2608.13524) reports its results with
    K=64 candidates per position and W=12 supertree width, so 8 puts us an order of
    magnitude below the configuration its numbers come from. Exposed so that gap is
    measurable rather than assumed harmless.
    """
    return max(2, envs.SGLANG_DFLASH_TFM_EXPAND_WIDTH.get())


def weaver_tree_batch_expand_width(tree_budget: Optional[int] = None) -> int:
    """Weaver expansion batch width: one Weaver call expands this many nodes.

    Scales with the tree budget so a tree of B nodes takes ~B/16 batched calls.
    """
    if tree_budget is None:
        return WEAVER_TREE_BATCH_EXPAND_WIDTH
    budget = int(tree_budget)
    if budget <= 0:
        return 1
    expand_unit = max(1, envs.SGLANG_DFLASH_TFM_EXPAND_UNIT.get())
    return max(
        1,
        (budget + expand_unit - 1) // expand_unit,
    )


def _weaver_indexed_attention(
    q: torch.Tensor,
    current_keys: torch.Tensor,
    current_values: torch.Tensor,
    external_keys: torch.Tensor,
    external_values: torch.Tensor,
    external_mask: torch.Tensor,
    node_keys: torch.Tensor,
    node_values: torch.Tensor,
    parent_ancestors: torch.Tensor,
    row_batch_indices: torch.Tensor,
    position_ids: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    if triton is None or q.device.type != "cuda":
        raise RuntimeError("indexed weaver attention requires Triton on CUDA.")
    rows, num_heads, head_dim = q.shape
    prefix = external_keys.shape[1]
    depth = parent_ancestors.shape[1]
    num_nodes = node_keys.shape[1]
    num_layers = node_keys.shape[2]
    out = torch.empty_like(q)
    block_ctx = triton.next_power_of_2(int(prefix + depth))
    block_d = triton.next_power_of_2(int(head_dim))
    _weaver_indexed_attention_kernel[(int(rows), int(num_heads))](
        q,
        current_keys,
        current_values,
        external_keys,
        external_values,
        external_mask,
        node_keys,
        node_values,
        parent_ancestors,
        row_batch_indices,
        position_ids,
        out,
        LAYER=int(layer_index),
        PREFIX=int(prefix),
        DEPTH=int(depth),
        NUM_NODES=int(num_nodes),
        NUM_LAYERS=int(num_layers),
        NUM_HEADS=int(num_heads),
        HEAD_DIM=int(head_dim),
        BLOCK_CTX=int(block_ctx),
        BLOCK_D=int(block_d),
    )
    return out


def _weaver_chain_attention(
    q: torch.Tensor,
    current_keys: torch.Tensor,
    current_values: torch.Tensor,
    external_keys: torch.Tensor,
    external_values: torch.Tensor,
    external_mask: torch.Tensor,
    chain_keys: torch.Tensor,
    chain_values: torch.Tensor,
    position_ids: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    if triton is None or q.device.type != "cuda":
        raise RuntimeError("chain weaver attention requires Triton on CUDA.")
    rows, num_heads, head_dim = q.shape
    prefix = external_keys.shape[1]
    depth = chain_keys.shape[1]
    num_layers = chain_keys.shape[2]
    out = torch.empty_like(q)
    block_ctx = triton.next_power_of_2(int(prefix + depth))
    block_d = triton.next_power_of_2(int(head_dim))
    _weaver_chain_attention_kernel[(int(rows), int(num_heads))](
        q,
        current_keys,
        current_values,
        external_keys,
        external_values,
        external_mask,
        chain_keys,
        chain_values,
        position_ids,
        out,
        LAYER=int(layer_index),
        PREFIX=int(prefix),
        DEPTH=int(depth),
        NUM_LAYERS=int(num_layers),
        NUM_HEADS=int(num_heads),
        HEAD_DIM=int(head_dim),
        BLOCK_CTX=int(block_ctx),
        BLOCK_D=int(block_d),
    )
    return out


TREE_ATTENTION_BACKENDS = frozenset(
    {
        "AiterAttnBackend",
        "FlashAttentionBackend",
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "TritonAttnBackend",
        "WaveAttnBackend",
        "XPUAttentionBackend",
    }
)


def _tree_attention_backend(attn_backend):
    backend = attn_backend
    for _ in range(8):
        select_backend = getattr(backend, "_select_backend", None)
        if select_backend is not None:
            selected_backend = select_backend(ForwardMode.TARGET_VERIFY)
            if selected_backend is not backend:
                backend = selected_backend
                continue

        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    return backend


def _tree_attention_backend_name(attn_backend) -> str:
    backend = _tree_attention_backend(attn_backend)
    return type(backend).__name__


def require_tree_attention_support(attn_backend) -> None:
    backend_name = _tree_attention_backend_name(attn_backend)
    if backend_name not in TREE_ATTENTION_BACKENDS:
        raise RuntimeError(
            "DFLASH_TFM requires TreeAttention custom-mask support, "
            f"but the selected target-verify attention backend is {backend_name}. "
            "Use a backend with speculative tree custom_mask support, such as "
            "triton, flashinfer, fa3/flashattention, aiter, or wave. "
            "For trtllm_mha decode, use a split backend with "
            "--prefill-attention-backend flashinfer and "
            "--speculative-attention-mode prefill."
        )


class SplitHiddenStates(msgspec.Struct):
    target_hidden: torch.Tensor
    output_norm: torch.Tensor


def split_dflash_tfm_hidden(
    hidden_states: torch.Tensor, hidden_size: int
) -> SplitHiddenStates:
    if hidden_states is None:
        raise RuntimeError("DFlash+Weaver requires captured target hidden states.")
    hidden_size = int(hidden_size)
    if hidden_states.shape[-1] <= hidden_size:
        raise RuntimeError(
            "DFlash+Weaver expected concatenated DFlash aux hidden and final hidden, "
            f"got feature_dim={hidden_states.shape[-1]}, hidden_size={hidden_size}."
        )
    return SplitHiddenStates(
        target_hidden=hidden_states[..., :-hidden_size].contiguous(),
        output_norm=hidden_states[..., -hidden_size:].contiguous(),
    )


def _last_extend_indices(
    extend_lens: torch.Tensor | List[int], device: torch.device
) -> torch.Tensor:
    if not isinstance(extend_lens, torch.Tensor):
        extend_lens = torch.tensor(extend_lens, dtype=torch.int64, device=device)
    else:
        extend_lens = extend_lens.to(device=device, dtype=torch.int64)
    return torch.cumsum(extend_lens, dim=0) - 1


class DFlashTfmDraftInput(DFlashDraftInputV2):
    output_norm: torch.Tensor

    def __init__(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        output_norm: torch.Tensor,
        committed_seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        bs = int(new_seq_lens.numel())
        device = bonus_tokens.device
        super().__init__(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
        )
        self.output_norm = output_norm
        self.committed_seq_lens_cpu = committed_seq_lens_cpu

    @classmethod
    def create_idle_input(
        cls, device: torch.device, output_norm_dim: int
    ) -> "DFlashTfmDraftInput":
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            output_norm=torch.empty(
                (0, int(output_norm_dim)), device=device, dtype=torch.float16
            ),
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        super().filter_batch(new_indices, has_been_filtered=has_been_filtered)
        self.output_norm = self.output_norm[new_indices]
        if self.committed_seq_lens_cpu is not None:
            self.committed_seq_lens_cpu = self.committed_seq_lens_cpu[
                new_indices.cpu()
            ]

    def merge_batch(self, spec_info: "DFlashTfmDraftInput"):
        super().merge_batch(spec_info)
        self.output_norm = torch.cat([self.output_norm, spec_info.output_norm], dim=0)
        if self.committed_seq_lens_cpu is not None:
            assert spec_info.committed_seq_lens_cpu is not None
            self.committed_seq_lens_cpu = torch.cat(
                [self.committed_seq_lens_cpu, spec_info.committed_seq_lens_cpu]
            )
        elif spec_info.committed_seq_lens_cpu is not None:
            self.committed_seq_lens_cpu = spec_info.committed_seq_lens_cpu


class WeaverRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (y * self.weight + self.bias).to(dtype=x.dtype)


class WeaverBlock(nn.Module):
    def __init__(self, d_rank: int, num_heads: int, mlp_dim: int):
        super().__init__()
        if d_rank % num_heads != 0:
            raise ValueError("d_rank must be divisible by num_heads")
        self.d_rank = int(d_rank)
        self.num_heads = int(num_heads)
        self.head_dim = int(d_rank // num_heads)
        self.norm_attn = WeaverRMSNorm(d_rank)
        self.qkv_proj = nn.Linear(d_rank, 3 * d_rank, bias=False)
        self.o_proj = nn.Linear(d_rank, d_rank, bias=False)
        self.norm_mlp = WeaverRMSNorm(d_rank)
        self.fc1 = nn.Linear(d_rank, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, d_rank)

    def forward(
        self,
        x: torch.Tensor,
        token_attention_mask: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, steps, _ = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv_proj(h).view(
            rows, steps, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        scale = self.head_dim**-0.5
        ext_scores = torch.einsum("rshd,rsphd->rhsp", q, external_keys) * scale
        tok_scores = torch.einsum("rshd,rthd->rhst", q, k) * scale
        ext_scores = ext_scores.masked_fill(~external_mask[:, None], -torch.inf)
        tok_scores = tok_scores.masked_fill(~token_attention_mask[:, None], -torch.inf)
        scores = torch.cat([ext_scores, tok_scores], dim=-1)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        prefix = external_keys.shape[2]
        ext_y = torch.einsum(
            "rhsp,rsphd->rshd", attn[:, :, :, :prefix], external_values
        )
        tok_y = torch.einsum("rhst,rthd->rshd", attn[:, :, :, prefix:], v)
        x = x + self.o_proj((ext_y + tok_y).reshape(rows, steps, self.d_rank))
        x = x + self.fc2(F.gelu(self.fc1(self.norm_mlp(x))))
        return x, k, v

    def forward_indexed(
        self,
        x: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        node_keys: torch.Tensor,
        node_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
        position_ids: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, steps, _ = x.shape
        if steps != 1:
            raise RuntimeError("indexed weaver step requires a single token step.")
        h = self.norm_attn(x)
        qkv = self.qkv_proj(h).view(
            rows, steps, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.squeeze(1).contiguous()
        k = k.squeeze(1).contiguous()
        v = v.squeeze(1).contiguous()
        y = _weaver_indexed_attention(
            q,
            k,
            v,
            external_keys,
            external_values,
            external_mask,
            node_keys,
            node_values,
            parent_ancestors,
            row_batch_indices,
            position_ids,
            layer_index,
        )
        x = x + self.o_proj(y.reshape(rows, steps, self.d_rank))
        x = x + self.fc2(F.gelu(self.fc1(self.norm_mlp(x))))
        return x, k, v

    def forward_chain(
        self,
        x: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        chain_keys: torch.Tensor,
        chain_values: torch.Tensor,
        position_ids: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, steps, _ = x.shape
        if steps != 1:
            raise RuntimeError("chain weaver step requires a single token step.")
        h = self.norm_attn(x)
        qkv = self.qkv_proj(h).view(
            rows, steps, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.squeeze(1).contiguous()
        k = k.squeeze(1).contiguous()
        v = v.squeeze(1).contiguous()
        y = _weaver_chain_attention(
            q,
            k,
            v,
            external_keys,
            external_values,
            external_mask,
            chain_keys,
            chain_values,
            position_ids,
            layer_index,
        )
        x = x + self.o_proj(y.reshape(rows, steps, self.d_rank))
        x = x + self.fc2(F.gelu(self.fc1(self.norm_mlp(x))))
        return x, k, v


class Weaver(nn.Module):
    ENCODER_GLOBAL_PROMPT = 3
    SCORE_SIMPLE = 4

    def __init__(
        self,
        *,
        d_model: int,
        d_embed: int,
        d_rank: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        K: int,
        candidate_pool_size: int,
        encoder_mode: int = ENCODER_GLOBAL_PROMPT,
        score_head: int = SCORE_SIMPLE,
    ):
        super().__init__()
        if int(encoder_mode) != self.ENCODER_GLOBAL_PROMPT:
            raise ValueError(
                "DFlash+Weaver MVP supports encoder_mode=global_prompt only."
            )
        if int(score_head) != self.SCORE_SIMPLE:
            raise ValueError(
                "DFlash+Weaver MVP supports score_head=simple_score only."
            )
        self.d_model = int(d_model)
        self.d_embed = int(d_embed)
        self.d_rank = int(d_rank)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.K = int(K)
        self.candidate_pool_size = int(candidate_pool_size)
        self.output_norm = WeaverRMSNorm(d_model)
        self.embed_norm = WeaverRMSNorm(d_embed)
        self.token_in = nn.Linear(d_embed, d_rank)
        self.proposal_in = nn.Linear(d_model, d_rank)
        self.blocks = nn.ModuleList(
            [WeaverBlock(d_rank, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.out_norm = WeaverRMSNorm(d_rank)
        self.lm_head_query_in = nn.Linear(d_rank, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(K, d_rank))

    @staticmethod
    def _migrate_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        migrated = dict(state_dict)
        q_suffix = "q_proj.weight"
        prefixes = [
            key[: -len(q_suffix)] for key in migrated.keys() if key.endswith(q_suffix)
        ]
        for prefix in prefixes:
            q_key = f"{prefix}q_proj.weight"
            k_key = f"{prefix}k_proj.weight"
            v_key = f"{prefix}v_proj.weight"
            qkv_key = f"{prefix}qkv_proj.weight"
            if qkv_key not in migrated:
                migrated[qkv_key] = torch.cat(
                    [migrated[q_key], migrated[k_key], migrated[v_key]], dim=0
                )
            migrated.pop(q_key, None)
            migrated.pop(k_key, None)
            migrated.pop(v_key, None)
        return migrated

    @classmethod
    def load(
        cls, path: str, *, device: torch.device, dtype: torch.dtype
    ) -> "Weaver":
        payload = torch.load(path, map_location=device)
        if (
            not isinstance(payload, dict)
            or "config" not in payload
            or "state_dict" not in payload
        ):
            raise ValueError(
                "Weaver checkpoint must be a torch file "
                "containing {'config': ..., 'state_dict': ...}. "
                "JAX/Equinox conversion is intentionally a separate final step."
            )
        model = cls(**payload["config"]).to(device=device, dtype=dtype)
        model.load_state_dict(
            cls._migrate_state_dict(payload["state_dict"]), strict=True
        )
        model.eval()
        return model

    def _token_project(
        self, token_ids: torch.Tensor, token_embed: torch.Tensor
    ) -> torch.Tensor:
        token_ids = token_ids.clamp(min=0, max=token_embed.shape[0] - 1)
        return torch.index_select(token_embed, 0, token_ids.reshape(-1)).view(
            *token_ids.shape, token_embed.shape[-1]
        )

    def _prompt_tokens(
        self,
        output_norm_features: torch.Tensor,
        proposal_features: torch.Tensor,
    ) -> torch.Tensor:
        rows, steps, _ = proposal_features.shape
        first_output = self.output_norm(output_norm_features[:, :1].float()).to(
            dtype=proposal_features.dtype
        )
        output_token = self.proposal_in(first_output).reshape(rows, 1, self.d_rank)
        proposal = self.output_norm(proposal_features.float()).to(
            dtype=proposal_features.dtype
        )
        proposal_tokens = self.proposal_in(proposal)
        proposal_tokens = (
            proposal_tokens + self.pos_emb[:steps].to(dtype=proposal_tokens.dtype)[None]
        )
        return torch.cat([output_token, proposal_tokens], dim=1)

    def prompt_external_kv(
        self,
        output_norm_features: torch.Tensor,
        proposal_features: torch.Tensor,
        steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._prompt_tokens(output_norm_features, proposal_features)
        rows, prefix, _ = x.shape
        attention_mask = torch.ones(
            (rows, prefix, prefix), dtype=torch.bool, device=x.device
        ).tril()
        empty_keys = torch.empty(
            (rows, prefix, 0, self.num_heads, self.d_rank // self.num_heads),
            dtype=x.dtype,
            device=x.device,
        )
        empty_mask = torch.empty((rows, prefix, 0), dtype=torch.bool, device=x.device)
        key_layers = []
        value_layers = []
        for block in self.blocks:
            x, layer_keys, layer_values = block(
                x,
                attention_mask,
                empty_keys,
                empty_keys,
                empty_mask,
            )
            key_layers.append(layer_keys)
            value_layers.append(layer_values)
        keys = torch.stack(key_layers)
        values = torch.stack(value_layers)
        return (
            keys,
            values,
            torch.ones((rows, prefix), dtype=torch.bool, device=x.device),
        )

    def step_indexed(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        node_keys: torch.Tensor,
        node_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = parent_ancestors.shape[1]
        x = self._token_project(token_ids[:, None], token_embed)
        position_ids = position_ids.clamp(min=0, max=depth - 1)
        pos_emb_ids = position_ids.clamp(max=self.K - 1)
        pos_emb = torch.index_select(self.pos_emb, 0, pos_emb_ids.reshape(-1)).view(
            pos_emb_ids.shape[0], self.d_rank
        )
        x = x + pos_emb[:, None].to(dtype=x.dtype)
        current_key_layers = []
        current_value_layers = []
        for layer_index, block in enumerate(self.blocks):
            x, layer_keys, layer_values = block.forward_indexed(
                x,
                external_keys[layer_index],
                external_values[layer_index],
                external_mask,
                node_keys,
                node_values,
                parent_ancestors,
                row_batch_indices,
                position_ids,
                layer_index,
            )
            current_key_layers.append(layer_keys)
            current_value_layers.append(layer_values)
        query = self.out_norm(x).to(dtype=candidate_weights.dtype).squeeze(1)
        residual = (
            torch.matmul(candidate_weights, query[:, :, None]).squeeze(-1).float()
        )
        logits = candidate_scores.float() + residual
        logits = logits.masked_fill(candidate_ids < 0, -torch.inf)
        return logits, torch.stack(current_key_layers), torch.stack(current_value_layers)

    def step_chain(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        chain_keys: torch.Tensor,
        chain_values: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = chain_keys.shape[1]
        x = self._token_project(token_ids[:, None], token_embed)
        position_ids = position_ids.clamp(min=0, max=depth - 1)
        pos_emb_ids = position_ids.clamp(max=self.K - 1)
        pos_emb = torch.index_select(self.pos_emb, 0, pos_emb_ids.reshape(-1)).view(
            pos_emb_ids.shape[0], self.d_rank
        )
        x = x + pos_emb[:, None].to(dtype=x.dtype)
        current_key_layers = []
        current_value_layers = []
        for layer_index, block in enumerate(self.blocks):
            x, layer_keys, layer_values = block.forward_chain(
                x,
                external_keys[layer_index],
                external_values[layer_index],
                external_mask,
                chain_keys,
                chain_values,
                position_ids,
                layer_index,
            )
            current_key_layers.append(layer_keys)
            current_value_layers.append(layer_values)
        query = self.out_norm(x).to(dtype=candidate_weights.dtype).squeeze(1)
        residual = (
            torch.matmul(candidate_weights, query[:, :, None]).squeeze(-1).float()
        )
        logits = candidate_scores.float() + residual
        logits = logits.masked_fill(candidate_ids < 0, -torch.inf)
        return logits, torch.stack(current_key_layers), torch.stack(current_value_layers)


class WeaverTree(msgspec.Struct):
    draft_tokens: torch.Tensor
    parent_indices: torch.Tensor
    depths: torch.Tensor
    node_mask: torch.Tensor
    draft_logprobs: torch.Tensor
    # True for the one child per node drawn from the residual draft distribution.
    # Verification needs to tell it apart from its deterministic top-K siblings.
    is_sampled: torch.Tensor
    # Per-node drafter pool and the residual distribution ms it was sampled from.
    pool_ids: torch.Tensor
    pool_ms: torch.Tensor


class WeaverTreeCudaGraph(msgspec.Struct):
    graph: torch.cuda.CUDAGraph
    root_ids: torch.Tensor
    output_norm: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_weights: torch.Tensor
    candidate_scores: torch.Tensor
    proposal_features: torch.Tensor
    tree: WeaverTree


class WeaverChainGraphSamplingInfo(msgspec.Struct):
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    is_all_greedy: bool
    need_top_p_sampling: bool
    need_top_k_sampling: bool


class WeaverChainCudaGraph(msgspec.Struct):
    graph: torch.cuda.CUDAGraph
    root_ids: torch.Tensor
    output_norm: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_weights: torch.Tensor
    candidate_scores: torch.Tensor
    proposal_features: torch.Tensor
    draft_tokens: torch.Tensor
    proposal_uniforms: Optional[torch.Tensor] = None
    proposal_tokens: Optional[torch.Tensor] = None
    proposal_probs: Optional[torch.Tensor] = None
    sampling_info: Optional[WeaverChainGraphSamplingInfo] = None


class WeaverChain(msgspec.Struct):
    draft_tokens: torch.Tensor
    proposal_tokens: Optional[torch.Tensor] = None
    proposal_probs: Optional[torch.Tensor] = None


def build_tree_metadata(
    *,
    draft_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    node_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    max_depth: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, num_nodes = draft_tokens.shape
    device = draft_tokens.device
    if device.type != "cuda":
        raise RuntimeError("tree metadata construction requires CUDA.")
    node_mask = node_mask.to(dtype=torch.bool, device=device)
    parent_indices = parent_indices.to(dtype=torch.long, device=device)

    retrieve_index = torch.arange(
        bs * num_nodes, dtype=torch.int64, device=device
    ).view(bs, num_nodes)
    retrieve_next_token = torch.empty((bs, num_nodes), dtype=torch.int64, device=device)
    retrieve_next_sibling = torch.empty_like(retrieve_next_token)
    positions = (seq_lens[:, None].to(torch.int64) + depths.to(torch.int64)).reshape(-1)

    prefix_lens = seq_lens.to(device=device, dtype=torch.int64)
    mask_sizes = num_nodes * (prefix_lens + num_nodes)
    mask_offsets = torch.empty((bs,), dtype=torch.int64, device=device)
    mask_offsets[0] = 0
    mask_offsets[1:] = torch.cumsum(mask_sizes[:-1], dim=0)

    prefix_lens_cpu = seq_lens_cpu.to(dtype=torch.int64)
    total_mask_size = int((num_nodes * (prefix_lens_cpu + num_nodes)).sum().item())
    custom_mask = torch.empty(total_mask_size, dtype=torch.bool, device=device)
    custom_mask.fill_(True)

    chain_steps = num_nodes if max_depth is None else min(num_nodes, int(max_depth) + 1)
    block_n = triton.next_power_of_2(int(num_nodes))
    _tree_metadata_parent_chain_kernel[(bs, num_nodes)](
        parent_indices,
        node_mask,
        prefix_lens,
        mask_offsets,
        custom_mask,
        retrieve_next_token,
        retrieve_next_sibling,
        NUM_NODES=int(num_nodes),
        CHAIN_STEPS=int(chain_steps),
        BLOCK_N=int(block_n),
    )
    return (
        custom_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def _traversal_verify_target_probs(
    *,
    candidates: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    node_mask: torch.Tensor,
    draft_logprobs: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    is_sampled: Optional[torch.Tensor] = None,
    pool_ids: Optional[torch.Tensor] = None,
    pool_ms: Optional[torch.Tensor] = None,
    univer_ok: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not candidates.is_cuda:
        raise RuntimeError("DFLASH_TFM traversal verification requires CUDA.")
    if candidates.dim() != 2:
        raise RuntimeError(
            f"traversal candidates must be rank-2, got {candidates.dim()}."
        )
    bs, num_nodes = candidates.shape
    if target_probs.shape[:2] != (bs, num_nodes):
        raise RuntimeError(
            "target_probs shape must start with candidates.shape, "
            f"got target_probs={tuple(target_probs.shape)}, candidates={tuple(candidates.shape)}."
        )
    target_probs = target_probs.contiguous()
    parent_indices = parent_indices.to(device=candidates.device, dtype=torch.int64)
    depths = depths.to(device=candidates.device, dtype=torch.int64)
    node_mask = node_mask.to(device=candidates.device, dtype=torch.bool)
    draft_logprobs = draft_logprobs.to(device=candidates.device, dtype=torch.float32)
    uniform_samples = uniform_samples.to(device=candidates.device, dtype=torch.float32)
    if uniform_samples.shape != (bs, num_nodes):
        raise RuntimeError(
            "uniform_samples shape mismatch for traversal verification: "
            f"expected {(bs, num_nodes)}, got {tuple(uniform_samples.shape)}."
        )

    predict = torch.empty((bs * num_nodes,), dtype=torch.int32, device=candidates.device)
    accept_index = torch.empty(
        (bs, num_nodes), dtype=torch.int32, device=candidates.device
    )
    num_correct = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
    accept_leaf = torch.empty((bs,), dtype=torch.int64, device=candidates.device)
    block_n = triton.next_power_of_2(int(num_nodes))
    # Never on a greedy tree: it has no sampled child, so UniVer's p(u_m) term is
    # undefined and the cascade is already exactly lossless there.
    # UniVer is REQUIRED, not optional, whenever the tree carries deterministic
    # children and verification is stochastic. The cascade's acceptance identity
    # min(1, cur_p*q/s) is lossless only for candidates DRAWN with probability s
    # (Traversal Verification, arXiv 2505.12398, Definition 3.1); applied to
    # top-K children it emits the drafter's argmax and measures TV 0.55-0.73.
    # Leaving that behind an opt-in knob meant the shipped default was non-lossless
    # at T>0 while every benchmark script set the knob -- i.e. the measurements were
    # fine and the artifact was not.
    use_univer = (
        univer_ok
        and is_sampled is not None
        and pool_ids is not None
        and pool_ms is not None
    )
    if not getattr(_traversal_verify_target_probs, "_logged", False):
        _traversal_verify_target_probs._logged = True
        # Emitted once per process. An env var that never reached the container is
        # otherwise indistinguishable from a null result, and this path decides
        # whether the run is lossless at all.
        logger.info(
            "DFLASH_TFM verify rule: %s (stochastic=%s tensors=%s)",
            "UniVer" if use_univer else "cascade (greedy: exactly lossless)",
            univer_ok,
            is_sampled is not None and pool_ids is not None and pool_ms is not None,
        )
    if univer_ok and not use_univer:
        # Stochastic verification with the cascade is not lossless on this tree.
        # Fail loudly rather than silently emitting the drafter's argmax.
        raise RuntimeError(
            "DFLASH_TFM: stochastic verification requires the UniVer path, but the "
            "per-node tensors are missing (is_sampled/pool_ids/pool_ms). The "
            "cascade is only lossless for candidates drawn from the draft "
            "distribution; on a top-K tree it emits the drafter's argmax."
        )
    if use_univer:
        vocab = int(target_probs.shape[-1])
        pool_size = int(pool_ids.shape[-1])
        residual = torch.zeros(
            (bs, vocab), dtype=torch.float32, device=candidates.device
        )
        residual_valid = torch.zeros(
            (bs,), dtype=torch.int32, device=candidates.device
        )
        _univer_verify_kernel[(int(bs),)](
            candidates.to(torch.int64),
            parent_indices,
            depths,
            node_mask,
            draft_logprobs,
            is_sampled.to(device=candidates.device, dtype=torch.bool),
            pool_ids.to(device=candidates.device, dtype=torch.int64),
            pool_ms.to(device=candidates.device, dtype=torch.float32),
            target_probs,
            uniform_samples,
            predict,
            accept_index,
            num_correct,
            accept_leaf,
            residual,
            residual_valid,
            NUM_NODES=int(num_nodes),
            VOCAB_SIZE=vocab,
            POOL_SIZE=pool_size,
            BLOCK_N=int(block_n),
            BLOCK_POOL=int(triton.next_power_of_2(pool_size)),
            BLOCK_V=1024,
            num_warps=8,
        )
        row_ids = torch.arange(bs, dtype=torch.long, device=candidates.device)
        dist = target_probs[row_ids, accept_leaf]
        # An interior node fired from its residual, not from q(.|v); a leaf still
        # draws its bonus token from the target.
        rv = residual_valid.bool()
        rs = residual.sum(dim=1, keepdim=True)
        dist = torch.where(
            (rv[:, None]) & (rs > 0), residual / rs.clamp_min(1e-20), dist
        )
        bonus = torch.multinomial(dist, 1).squeeze(1)
        predict[row_ids * num_nodes + accept_leaf] = bonus.to(torch.int32)
        return predict, accept_index, num_correct, accept_leaf

    _weaver_traversal_verify_kernel[(int(bs),)](
        candidates.to(torch.int64),
        parent_indices,
        depths,
        node_mask,
        draft_logprobs,
        target_probs,
        uniform_samples,
        predict,
        accept_index,
        num_correct,
        accept_leaf,
        NUM_NODES=int(num_nodes),
        VOCAB_SIZE=int(target_probs.shape[-1]),
        BLOCK_N=int(block_n),
        num_warps=8,
    )
    row_ids = torch.arange(bs, dtype=torch.long, device=candidates.device)
    bonus = torch.multinomial(target_probs[row_ids, accept_leaf], 1).squeeze(1)
    predict[row_ids * num_nodes + accept_leaf] = bonus.to(torch.int32)
    return predict, accept_index, num_correct, accept_leaf


class DFlashTfmVerifyInput(DFlashVerifyInput):
    def __init__(
        self,
        *,
        draft_token: torch.Tensor,
        positions: torch.Tensor,
        draft_token_num: int,
        custom_mask: torch.Tensor,
        mask_seq_lens_cpu: Optional[torch.Tensor] = None,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        depths: torch.Tensor,
        parent_indices: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        draft_logprobs: Optional[torch.Tensor] = None,
        is_sampled: Optional[torch.Tensor] = None,
        pool_ids: Optional[torch.Tensor] = None,
        pool_ms: Optional[torch.Tensor] = None,
        univer_ok: bool = False,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL,
    ):
        super().__init__(
            draft_token=draft_token,
            positions=positions,
            draft_token_num=int(draft_token_num),
            topk=2,
            custom_mask=custom_mask,
            capture_hidden_mode=capture_hidden_mode,
        )
        self.retrieve_index = retrieve_index
        self.retrieve_next_token = retrieve_next_token
        self.retrieve_next_sibling = retrieve_next_sibling
        self.mask_seq_lens_cpu = mask_seq_lens_cpu
        self.mask_seq_lens_sum = (
            int(mask_seq_lens_cpu.sum().item())
            if mask_seq_lens_cpu is not None
            else None
        )
        self.depths = depths
        self.parent_indices = parent_indices
        self.node_mask = node_mask
        self.draft_logprobs = draft_logprobs
        self.is_sampled = is_sampled
        self.pool_ids = pool_ids
        self.pool_ms = pool_ms
        self.univer_ok = univer_ok
        # Tree-local slot of each request's last accepted node; populated by
        # verify() and consumed by the post-verify Mamba/GDN state commit.
        self.accept_leaf_slots: Optional[torch.Tensor] = None

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ) -> tuple[ForwardBatch, bool]:
        if not build_custom_mask or self.custom_mask is None:
            raise RuntimeError(
                "DFLASH_TFM requires TreeAttention custom_mask support; "
                "disabling or omitting the tree mask would change verification semantics."
            )
        batch.input_ids = self.draft_token
        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode
        if not batch.forward_mode.is_idle():
            end_offset = batch.seq_lens + int(self.draft_token_num)
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=end_offset,
                batch_size=batch.batch_size(),
                draft_token_num=int(self.draft_token_num),
                device=batch.device,
            )

        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)
        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.load_batch(
                verify_forward_batch
            )
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )
        return verify_forward_batch, can_run_cuda_graph

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
    ):
        # Weaver tree masks are laid out against the logical committed prefix
        # length. The spec-v2 page allocator may pass a larger host-side
        # planning/reserved sum for buffer sizing; using that value here would
        # make FlashInfer pad the tree mask as if the reserved KV tail belonged
        # to the attention prefix.
        if self.mask_seq_lens_sum is not None:
            paged_kernel_lens_sum = self.mask_seq_lens_sum
        return super().generate_attn_arg_prefill(
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            req_to_token,
            kv_start_idx,
        )

    def _verify_from_target_predict(self, target_predict: torch.Tensor, bs: int):
        candidates = self.draft_token.view(bs, self.draft_token_num)
        predict = torch.full(
            (bs * self.draft_token_num,),
            -1,
            dtype=torch.int32,
            device=candidates.device,
        )
        accept_index = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=candidates.device
        )
        num_correct = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
        if not (is_cuda() or is_musa()):
            for b in range(bs):
                last = int(self.retrieve_index[b, 0].item())
                accept_index[b, 0] = last
                num_correct_drafts = 0
                cur = 0
                for _ in range(1, self.draft_token_num):
                    cur = int(self.retrieve_next_token[b, cur].item())
                    while cur != -1:
                        draft_index = int(self.retrieve_index[b, cur].item())
                        draft_token = int(candidates[b, cur].item())
                        target_token = int(target_predict.view(-1)[last].item())
                        if draft_token == target_token:
                            predict[last] = target_token
                            num_correct_drafts += 1
                            accept_index[b, num_correct_drafts] = draft_index
                            last = draft_index
                            break
                        cur = int(self.retrieve_next_sibling[b, cur].item())
                    if cur == -1:
                        break
                num_correct[b] = num_correct_drafts
                predict[last] = int(target_predict.view(-1)[last].item())
            return predict, accept_index, num_correct
        from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func

        verify_tree_greedy_func(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=num_correct,
            candidates=candidates,
            retrieve_index=self.retrieve_index,
            retrieve_next_token=self.retrieve_next_token,
            retrieve_next_sibling=self.retrieve_next_sibling,
            target_predict=target_predict,
        )
        return predict, accept_index, num_correct

    def _greedy_verify(self, logits_output: LogitsProcessorOutput, bs: int):
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, self.draft_token_num
        )
        return self._verify_from_target_predict(target_predict, bs)

    def _sampling_verify(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput, sampling_info
    ):
        bs = batch.batch_size()
        candidates = self.draft_token.view(bs, self.draft_token_num)
        if (
            self.parent_indices is None
            or self.node_mask is None
            or self.draft_logprobs is None
        ):
            raise RuntimeError(
                "DFLASH_TFM traversal verification requires tree parents, "
                "node mask, and draft log-probabilities."
            )
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, self.draft_token_num, dim=0
        )
        target_probs = F.softmax(
            logits_output.next_token_logits / expanded_temperature, dim=-1
        )
        if getattr(sampling_info, "need_top_k_sampling", True):
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.need_top_p_sampling:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
        if getattr(sampling_info, "need_min_p_sampling", False):
            # Losslessness is a statement about ONE distribution: the verifier must
            # score against exactly what the sampler would have drawn from. sampler.py
            # applies min_p (see Sampler.forward, need_min_p_sampling branch), so
            # omitting it here silently verifies against a different target -- the
            # request stays lossless w.r.t. a distribution nobody asked for.
            min_ps = torch.repeat_interleave(
                sampling_info.min_ps, self.draft_token_num, dim=0
            )
            thresh = min_ps.unsqueeze(-1) * target_probs.max(dim=-1, keepdim=True).values
            target_probs = torch.where(
                target_probs >= thresh, target_probs, 0.0
            )
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
        target_probs = target_probs.view(bs, self.draft_token_num, -1)
        predict, accept_index, num_correct, _ = _traversal_verify_target_probs(
            candidates=candidates.to(torch.int64),
            parent_indices=self.parent_indices,
            depths=self.depths,
            node_mask=self.node_mask,
            draft_logprobs=self.draft_logprobs,
            target_probs=target_probs,
            uniform_samples=torch.rand_like(candidates, dtype=torch.float32),
            is_sampled=self.is_sampled,
            pool_ids=self.pool_ids,
            pool_ms=self.pool_ms,
            univer_ok=self.univer_ok,
        )
        return predict, accept_index, num_correct

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
        hidden_size: int,
        token_to_kv_pool_allocator=None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
    ]:
        """Verify the Weaver tree and return spec-v2 style commit data.

        This method intentionally does not append to req.output_ids or update
        request-level speculative counters. The spec-v2 result processor owns
        output mutation from ``next_token_ids`` and ``accept_lens``. We still
        assign the accepted tree slots into the committed prefix and advance
        the batch-local sequence lengths so the draft KV materialization below
        can use the accepted target slots. Unaccepted slots stay in the DFlashV2
        over-allocation window and may be reused by the next decode step.
        """
        bs = batch.batch_size()
        sampling_info = batch.sampling_info
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits_output.next_token_logits,
            sampling_info=sampling_info,
            draft_token_num=self.draft_token_num,
        )
        if sampling_info is None or sampling_info.is_all_greedy:
            predict, accept_index, num_correct = self._greedy_verify(
                logits_output, bs
            )
        else:
            predict, accept_index, num_correct = self._sampling_verify(
                batch, logits_output, sampling_info
            )

        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        commit_lens_cpu: List[int] = []
        num_correct_cpu: List[int] = []
        out_tokens_cpu: List[List[int]] = []
        for row in accept_index_cpu:
            row_tokens: List[int] = []
            for idx in row:
                if idx == -1:
                    break
                row_tokens.append(int(predict_cpu[int(idx)]))
            if not row_tokens:
                raise RuntimeError(
                    "DFlash+Weaver verify produced an empty accept path."
                )
            commit_lens_cpu.append(len(row_tokens))
            num_correct_cpu.append(max(0, len(row_tokens) - 1))
            out_tokens_cpu.append(row_tokens)

        commit_lens = torch.tensor(
            commit_lens_cpu, dtype=torch.int32, device=batch.device
        )
        row_ids = torch.arange(bs, device=batch.device, dtype=torch.long)
        self.accept_leaf_slots = (
            accept_index[row_ids, commit_lens.to(torch.long) - 1].to(torch.long)
            - row_ids * self.draft_token_num
        )
        out_tokens = torch.zeros(
            (bs, self.draft_token_num), dtype=torch.int64, device=batch.device
        )
        for i, row_tokens in enumerate(out_tokens_cpu):
            out_tokens[i, : len(row_tokens)] = torch.tensor(
                row_tokens, dtype=torch.int64, device=batch.device
            )

        out_cache_loc = batch.out_cache_loc
        out_cache_loc_2d = out_cache_loc.view(bs, self.draft_token_num)
        if bs == 1:
            flat_accept = accept_index[0, : commit_lens_cpu[0]].to(torch.long)
        else:
            flat_accept = torch.cat(
                [
                    accept_index[i, :commit_len]
                    for i, commit_len in enumerate(commit_lens_cpu)
                ]
            ).to(torch.long)

        if page_size > 1:
            if token_to_kv_pool_allocator is None:
                raise RuntimeError(
                    "DFLASH_TFM page_size>1 commit requires target KV cache access."
                )
            dst_parts = []
            for i, commit_len in enumerate(commit_lens_cpu):
                if commit_len > 0:
                    dst_parts.append(out_cache_loc_2d[i, :commit_len])
                if commit_len < self.draft_token_num:
                    req_idx = batch.req_pool_indices[i].to(torch.long)
                    seq_len = int(batch.seq_lens_cpu[i].item())
                    batch.req_to_token_pool.req_to_token[
                        req_idx,
                        seq_len + commit_len : seq_len + self.draft_token_num,
                    ] = out_cache_loc_2d[i, commit_len : self.draft_token_num]
            compact_cache_loc = (
                torch.cat(dst_parts) if dst_parts else out_cache_loc.new_empty((0,))
            )
            accept_cache_loc = out_cache_loc[flat_accept]
            token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                compact_cache_loc, accept_cache_loc
            )
            batch.out_cache_loc = compact_cache_loc

        else:
            for i, row in enumerate(accept_index_cpu):
                accept_local = {
                    int(idx) - i * self.draft_token_num for idx in row if idx != -1
                }
                commit_len = commit_lens_cpu[i]
                if commit_len >= self.draft_token_num:
                    continue
                remaining_local = [
                    j
                    for j in range(self.draft_token_num)
                    if j not in accept_local
                ]
                req_idx = batch.req_pool_indices[i].to(torch.long)
                seq_len = int(batch.seq_lens_cpu[i].item())
                remaining_slots = out_cache_loc[
                    i * self.draft_token_num
                    + torch.tensor(
                        remaining_local, dtype=torch.long, device=batch.device
                    )
                ]
                batch.req_to_token_pool.req_to_token[
                    req_idx,
                    seq_len + commit_len : seq_len + self.draft_token_num,
                ] = remaining_slots
            batch.out_cache_loc = out_cache_loc[flat_accept]

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + commit_lens.to(batch.seq_lens.dtype),
            batch.out_cache_loc,
            bs,
        )
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        split = split_dflash_tfm_hidden(
            logits_output.hidden_states, hidden_size
        )
        target_hidden = split.target_hidden[flat_accept]
        target_positions = self.positions[flat_accept]
        output_norm = split.output_norm[flat_accept]
        terminal_offsets = torch.cumsum(commit_lens.to(torch.long), dim=0) - 1
        next_output_norm = output_norm[terminal_offsets]
        logits_output.hidden_states = None
        return (
            out_tokens,
            commit_lens,
            target_hidden,
            target_positions,
            next_output_norm,
            num_correct_cpu,
        )




# --- D-cut potential probe -------------------------------------------------
# Bole allocates a batch-wide node budget by draft probability, giving request i a
# variable count q_i instead of the uniform per-request budget used here. Whether
# that is worth implementing depends entirely on how much requests in one batch
# actually differ: if every request's marginal (lowest selected) score is similar,
# a batch-global top-N returns the uniform split and there is nothing to win.
#
# This probe records the prefix scores of selected nodes and reports the
# allocation a global top-N would have produced. Enabled by
# SGLANG_DFLASH_TFM_DCUT_PROBE; off by default and free when off.
_DCUT_STATE = {"scores": [], "calls": 0, "reported": 0}


def _dcut_probe_active() -> bool:
    return envs.SGLANG_DFLASH_TFM_DCUT_PROBE.get()


def _dcut_record(node_score: torch.Tensor, valid: torch.Tensor) -> None:
    _DCUT_STATE["scores"].append(
        torch.where(valid, node_score, torch.full_like(node_score, -float("inf")))
        .detach()
    )


def _dcut_report(bs: int, node_budget: int) -> None:
    """Compare uniform per-request allocation against a batch-global top-N."""
    chunks = _DCUT_STATE["scores"]
    _DCUT_STATE["scores"] = []
    if not chunks:
        return
    scores = torch.cat(chunks, dim=1)              # [bs, selected]
    total = bs * node_budget
    flat = scores.reshape(-1)
    finite = int((flat > -float("inf")).sum())
    if finite == 0:
        return
    take = min(total, finite)
    _, idx = torch.topk(flat, take)
    per_req = torch.bincount(idx // scores.shape[1], minlength=bs).float()
    # marginal score = lowest selected score per request under the uniform split
    marg = scores.masked_fill(scores == -float("inf"), float("inf")).min(dim=1).values
    marg = marg[marg < float("inf")]
    logger.info(
        "DFLASH_TFM D-cut probe: bs=%d uniform=%d/req | global top-N per-req "
        "min=%.1f max=%.1f mean=%.1f std=%.2f | reallocated=%.1f%% | "
        "marginal score spread=%.3f (min %.3f max %.3f)",
        bs, node_budget, float(per_req.min()), float(per_req.max()),
        float(per_req.mean()), float(per_req.std()),
        float((per_req - node_budget).abs().sum() / (2 * total) * 100),
        float(marg.max() - marg.min()) if marg.numel() else 0.0,
        float(marg.min()) if marg.numel() else 0.0,
        float(marg.max()) if marg.numel() else 0.0,
    )



def _resolve_tree_budget(server_args) -> int:
    """Tree budget, optionally derived from a batch-wide row target.

    Bole (arXiv 2608.01651) calibrates a batch-wide verification budget
    `B_ver(c) = max{N : T_ver(N|c) <= (1+eps) T_dec(c)}` where N is the TOTAL
    verified nodes across the batch, and reports +5.3-10.7% from that alone. Our
    budget is per-request with no total cap, so total nodes grow linearly with
    concurrency: at c64, budget 31 means 2048 verify tokens, which OOMs on
    activations at every memory fraction tried (0.75 / 0.72 / 0.70 / 0.68).

    Two policies, both resolved once at startup from max_running_requests --
    draft_token_num is baked into the CUDA graph capture and cannot vary per step
    without capturing a graph per size, and graph private pools are precisely what
    is scarce at high concurrency (8.62 GiB measured at c64).

    SGLANG_DFLASH_TFM_BUDGET_SCHEDULE ("batch:budget,..." , e.g. "1:95,8:31,32:31,
    64:16") picks the entry for the largest batch <= max_running_requests. This is
    the measured form and it is what Bole actually specifies: B_ver is calibrated
    per configuration bucket, not as one number.

    SGLANG_DFLASH_TFM_TARGET_ROWS caps batch * (budget + 1) at a constant instead.
    Kept for comparison, but our measurements say a CONSTANT row target is the
    wrong model: the optimal total rows GROWS with batch (96 at c1, 256 at c8,
    1024 at c32), so a target of 1024 would pick budget 127 at c8 where 31 measures
    best.

    Tree mode additionally requires num_draft_tokens > block_size, so the floor is
    block_size + 1 nodes; below that the engine silently falls back to chain
    verification, which reserves intermediate SSM state and does not fit at c64.
    """
    explicit = int(server_args.speculative_dflash_tfm_tree_budget or 128)
    batch = max(1, int(server_args.max_running_requests or 1))
    block_size = int(server_args.speculative_dflash_block_size or 16)
    # Tree mode needs num_draft_tokens > block_size; below that the engine falls
    # back to chain verification, which reserves intermediate SSM state and does
    # not fit at c64.
    floor = block_size

    schedule = envs.SGLANG_DFLASH_TFM_BUDGET_SCHEDULE.get().strip()
    if schedule:
        entries = []
        for part in schedule.split(","):
            k, _, v = part.partition(":")
            if not v:
                raise ValueError(
                    f"SGLANG_DFLASH_TFM_BUDGET_SCHEDULE entry {part!r} is not "
                    "batch:budget"
                )
            entries.append((int(k), int(v)))
        entries.sort()
        chosen = entries[0][1]
        for b, v in entries:
            if batch >= b:
                chosen = v
        budget = max(floor, min(explicit, chosen))
        logger.info(
            "DFLASH_TFM tree budget: %d from schedule %r at batch %d (%d rows)",
            budget, schedule, batch, batch * (budget + 1),
        )
        return budget

    target_rows = envs.SGLANG_DFLASH_TFM_TARGET_ROWS.get()
    if target_rows <= 0:
        return explicit
    budget = max(floor, min(explicit, max(floor + 1, target_rows // batch) - 1))
    logger.info(
        "DFLASH_TFM tree budget: %d (explicit %d, target_rows %d, batch %d "
        "-> %d rows)",
        budget, explicit, target_rows, batch, batch * (budget + 1),
    )
    return budget


def _raise_dynamo_recompile_limit(minimum: int) -> None:
    """Ensure Dynamo will tolerate `minimum` recompiles of a static-shape fn.

    Only ever raises the limit. The shape count here is bounded by
    max_batch_size x distinct expansion widths, so this trades a bounded amount
    of extra compile time at warmup for not dying at concurrency; it does not
    make the limit unbounded.
    """
    cfg = torch._dynamo.config
    for attr in ("recompile_limit", "cache_size_limit"):
        cur = getattr(cfg, attr, None)
        if isinstance(cur, int) and cur < minimum:
            setattr(cfg, attr, minimum)


class DFlashTfmWorker(DFlashWorkerV2):
    _knobs_logged = False
    _dartree_logged = False
    _dartree_calls = 0
    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: List[int], batch_size: int = 0
    ) -> None:
        """Spec-v2 result processor hook; Weaver is not adaptive yet."""
        pass

    def __init__(self, *args, **kwargs):
        server_args = args[0] if args else kwargs["server_args"]
        target_verify_tokens = server_args.speculative_num_draft_tokens
        if target_verify_tokens is None:
            target_verify_tokens = (
                int(server_args.speculative_dflash_tfm_tree_budget or 128) + 1
            )
        dflash_block_size_value = (
            server_args.speculative_dflash_block_size or target_verify_tokens
        )
        if dflash_block_size_value is None:
            raise ValueError(
                "DFLASH_TFM requires a DFlash block size. "
                "Run the speculative arg hook or set --speculative-dflash-block-size."
            )
        dflash_block_size = int(dflash_block_size_value)
        server_args.speculative_num_draft_tokens = dflash_block_size
        try:
            super().__init__(*args, **kwargs)
        finally:
            server_args.speculative_num_draft_tokens = target_verify_tokens
        path = self.server_args.speculative_dflash_tfm_path
        if path is None:
            raise ValueError(
                "DFLASH_TFM requires --speculative-dflash-tfm-path."
            )
        dtype = getattr(
            self.target_worker.model_runner.model_config, "dtype", torch.bfloat16
        )
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        self.weaver = Weaver.load(
            path,
            device=self.device,
            dtype=dtype,
        )
        self.tree_budget = _resolve_tree_budget(self.server_args)
        requested_pool_size = int(
            self.server_args.speculative_dflash_tfm_candidate_pool_size
            or self.weaver.candidate_pool_size
        )
        if requested_pool_size <= 0:
            raise ValueError(
                "DFLASH_TFM candidate pool size must be positive, "
                f"got {requested_pool_size}."
            )
        self.candidate_pool_size = min(
            requested_pool_size, int(self.weaver.candidate_pool_size)
        )
        if get_tp_group().world_size != 1:
            raise NotImplementedError(
                "DFLASH_TFM MVP supports tensor_parallel_size=1 only."
            )
        self.hidden_size = int(self.target_worker.model_runner.model_config.hidden_size)
        self._weaver_residual_lm_head_cache: Optional[torch.Tensor] = None
        self._weaver_residual_lm_head_cache_key: Optional[tuple[object, ...]] = None
        self._weaver_token_embed_cache: Optional[torch.Tensor] = None
        self._weaver_token_embed_cache_key: Optional[tuple[object, ...]] = None
        self._weaver_tree_cuda_graphs: dict[
            tuple[object, ...], WeaverTreeCudaGraph
        ] = {}
        self._weaver_chain_cuda_graphs: dict[
            tuple[object, ...], WeaverChainCudaGraph
        ] = {}
        self.target_verify_tokens = int(
            self.server_args.speculative_num_draft_tokens or self.block_size
        )
        self.use_chain_verify = self.target_verify_tokens <= int(self.block_size)

    def init_attention_backends(self):
        if self.target_verify_tokens <= int(self.block_size):
            backend_name = _tree_attention_backend_name(
                self.target_worker.model_runner.attn_backend
            )
            if backend_name not in TREE_ATTENTION_BACKENDS:
                self.use_chain_verify = True
        if not self.use_chain_verify:
            require_tree_attention_support(self.target_worker.model_runner.attn_backend)
        super().init_attention_backends()

    def _target_embedding_and_lm_head(self):
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH_TFM requires vocab-parallel target lm_head."
            )
        if not hasattr(embed_module, "weight"):
            raise RuntimeError(
                "DFLASH_TFM requires target input embedding weight."
            )
        return embed_module, lm_head

    def _weaver_token_embed(self, embed_module) -> torch.Tensor:
        weight = embed_module.weight
        norm_weight = self.weaver.embed_norm.weight
        norm_bias = self.weaver.embed_norm.bias
        projection_weight = self.weaver.token_in.weight
        projection_bias = self.weaver.token_in.bias
        key = (
            weight.data_ptr(),
            norm_weight.data_ptr(),
            norm_bias.data_ptr(),
            projection_weight.data_ptr(),
            projection_bias.data_ptr(),
            tuple(weight.shape),
            weight.dtype,
            projection_weight.dtype,
            weight.device,
            projection_weight.device,
            getattr(weight, "_version", 0),
            getattr(norm_weight, "_version", 0),
            getattr(norm_bias, "_version", 0),
            getattr(projection_weight, "_version", 0),
            getattr(projection_bias, "_version", 0),
        )
        if self._weaver_token_embed_cache_key != key:
            with torch.inference_mode():
                normalized = self.weaver.embed_norm(weight.float()).to(
                    dtype=weight.dtype
                )
                self._weaver_token_embed_cache = self.weaver.token_in(
                    normalized
                ).contiguous()
            self._weaver_token_embed_cache_key = key
        assert self._weaver_token_embed_cache is not None
        return self._weaver_token_embed_cache

    def _weaver_residual_lm_head(self, lm_head) -> torch.Tensor:
        weight = lm_head.weight
        projection = self.weaver.lm_head_query_in.weight
        key = (
            weight.data_ptr(),
            projection.data_ptr(),
            tuple(weight.shape),
            tuple(projection.shape),
            weight.dtype,
            projection.dtype,
            weight.device,
            projection.device,
            getattr(weight, "_version", 0),
            getattr(projection, "_version", 0),
        )
        if self._weaver_residual_lm_head_cache_key != key:
            with torch.inference_mode():
                self._weaver_residual_lm_head_cache = torch.matmul(
                    weight.to(dtype=projection.dtype), projection
                ).contiguous()
            self._weaver_residual_lm_head_cache_key = key
        assert self._weaver_residual_lm_head_cache is not None
        return self._weaver_residual_lm_head_cache

    def _topk_from_lm_head(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shard = lm_head.shard_indices
        weight = lm_head.weight
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)
        hs = hidden_states.to(dtype=weight.dtype)
        if num_org > 0 and num_added == 0:
            logits = torch.matmul(hs, weight[:num_org].T).float()
            values, indices = torch.topk(logits, min(int(k), logits.shape[-1]), dim=-1)
            return values, indices.to(torch.long) + org_vocab_start

        logits_parts = []
        ids_parts = []
        if num_org > 0:
            logits_parts.append(torch.matmul(hs, weight[:num_org].T).float())
            ids_parts.append(
                torch.arange(
                    org_vocab_start,
                    org_vocab_start + num_org,
                    dtype=torch.long,
                    device=hs.device,
                )
            )
        if num_added > 0:
            added = weight[num_org_padded : num_org_padded + num_added]
            logits_parts.append(torch.matmul(hs, added.T).float())
            ids_parts.append(
                torch.arange(
                    added_vocab_start,
                    added_vocab_start + num_added,
                    dtype=torch.long,
                    device=hs.device,
                )
            )
        logits = torch.cat(logits_parts, dim=-1)
        ids = torch.cat(ids_parts, dim=0)
        _, indices = torch.topk(logits, min(int(k), logits.shape[-1]), dim=-1)
        values = torch.gather(logits, 1, indices)
        return values, ids[indices]

    def _weaver_indexed_step_compiled(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        node_keys: torch.Tensor,
        node_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            tuple(token_ids.shape),
            token_ids.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(external_keys.shape),
            external_keys.dtype,
            tuple(external_values.shape),
            external_values.dtype,
            tuple(external_mask.shape),
            tuple(position_ids.shape),
            tuple(node_keys.shape),
            node_keys.dtype,
            tuple(node_values.shape),
            node_values.dtype,
            tuple(parent_ancestors.shape),
            tuple(row_batch_indices.shape),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        compiled_steps = getattr(self, "_weaver_compiled_indexed_step_fns", None)
        if compiled_steps is None:
            compiled_steps = {}
            self._weaver_compiled_indexed_step_fns = compiled_steps
        compiled_step = compiled_steps.get(key)
        if compiled_step is None:
            def step_fn(
                token_ids,
                candidate_ids,
                candidate_weights,
                candidate_scores,
                external_keys,
                external_values,
                external_mask,
                position_ids,
                node_keys,
                node_values,
                parent_ancestors,
                row_batch_indices,
                token_embed,
            ):
                return self.weaver.step_indexed(
                    token_ids=token_ids,
                    candidate_ids=candidate_ids,
                    candidate_weights=candidate_weights,
                    candidate_scores=candidate_scores,
                    external_keys=external_keys,
                    external_values=external_values,
                    external_mask=external_mask,
                    position_ids=position_ids,
                    node_keys=node_keys,
                    node_values=node_values,
                    parent_ancestors=parent_ancestors,
                    row_batch_indices=row_batch_indices,
                    token_embed=token_embed,
                )

            # dynamic=False compiles one graph per input shape, and this
            # function sees batch_size x expansion_width distinct shapes: the
            # tree CUDA graphs capture every batch size up to
            # cuda_graph_max_bs_decode, and the expansion loop's final iteration
            # is narrower than the rest whenever the budget is not a multiple of
            # the batch expand width. At c8 with budget 31 that is 8 x 2 = 16
            # shapes against Dynamo's default recompile_limit of 8, and exceeding
            # it raises inside the scheduler and kills the server mid-run.
            # Budget 64 happens to use a single width, which is why c8 worked
            # there and nowhere else.
            _raise_dynamo_recompile_limit(256)
            compiled_step = torch.compile(
                step_fn,
                fullgraph=True,
                dynamic=False,
                options={
                    "triton.cudagraphs": False,
                    "emulate_precision_casts": True,
                },
            )
            compiled_steps[key] = compiled_step
        return compiled_step(
            token_ids,
            candidate_ids,
            candidate_weights,
            candidate_scores,
            external_keys,
            external_values,
            external_mask,
            position_ids,
            node_keys,
            node_values,
            parent_ancestors,
            row_batch_indices,
            token_embed,
        )

    def _weaver_chain_step_compiled(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        chain_keys: torch.Tensor,
        chain_values: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            tuple(token_ids.shape),
            token_ids.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(external_keys.shape),
            external_keys.dtype,
            tuple(external_values.shape),
            external_values.dtype,
            tuple(external_mask.shape),
            tuple(position_ids.shape),
            tuple(chain_keys.shape),
            chain_keys.dtype,
            tuple(chain_values.shape),
            chain_values.dtype,
            token_embed.data_ptr(),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        compiled_steps = getattr(self, "_weaver_compiled_chain_step_fns", None)
        if compiled_steps is None:
            compiled_steps = {}
            self._weaver_compiled_chain_step_fns = compiled_steps
        compiled_step = compiled_steps.get(key)
        if compiled_step is None:
            def step_fn(
                token_ids,
                candidate_ids,
                candidate_weights,
                candidate_scores,
                external_keys,
                external_values,
                external_mask,
                position_ids,
                chain_keys,
                chain_values,
                token_embed,
            ):
                return self.weaver.step_chain(
                    token_ids=token_ids,
                    candidate_ids=candidate_ids,
                    candidate_weights=candidate_weights,
                    candidate_scores=candidate_scores,
                    external_keys=external_keys,
                    external_values=external_values,
                    external_mask=external_mask,
                    position_ids=position_ids,
                    chain_keys=chain_keys,
                    chain_values=chain_values,
                    token_embed=token_embed,
                )

            # Same bounded-shape exposure as the tree path above; the chain
            # path is what died first at c8 (budget 15).
            _raise_dynamo_recompile_limit(256)
            compiled_step = torch.compile(
                step_fn,
                fullgraph=True,
                dynamic=False,
                options={
                    "triton.cudagraphs": False,
                    "emulate_precision_casts": True,
                },
            )
            compiled_steps[key] = compiled_step
        return compiled_step(
            token_ids,
            candidate_ids,
            candidate_weights,
            candidate_scores,
            external_keys,
            external_values,
            external_mask,
            position_ids,
            chain_keys,
            chain_values,
            token_embed,
        )


    def _weaver_expand_for_dartree(
        self,
        *,
        token,
        node_depth,
        parent_ancestors,
        active,
        row_batch_indices,
        depth,
        candidate_ids_rows,
        candidate_weights_rows,
        candidate_scores_rows,
        external_keys,
        external_values,
        external_mask,
        node_keys,
        node_values,
        token_embed,
    ):
        """One Weaver expansion step over an arbitrary set of nodes.

        Same call as the best-first path makes, but decoupled from that path's
        contiguous-slot bookkeeping: DARTree expands a beam whose members sit at
        scattered candidate indices.
        """
        depth_index = node_depth.clamp(max=depth - 1)
        candidate_row_index = row_batch_indices * depth + depth_index
        row_candidate_ids = candidate_ids_rows[candidate_row_index]
        logits, current_keys, current_values = self._weaver_indexed_step_compiled(
            token_ids=torch.where(active, token, torch.zeros_like(token)),
            candidate_ids=row_candidate_ids,
            candidate_weights=candidate_weights_rows[candidate_row_index],
            candidate_scores=candidate_scores_rows[candidate_row_index],
            external_keys=external_keys,
            external_values=external_values,
            external_mask=external_mask,
            position_ids=depth_index,
            node_keys=node_keys,
            node_values=node_values,
            parent_ancestors=parent_ancestors.reshape(
                parent_ancestors.shape[0] * parent_ancestors.shape[1], depth
            ).contiguous(),
            row_batch_indices=row_batch_indices,
            token_embed=token_embed,
        )
        return logits.float(), row_candidate_ids, current_keys, current_values

    def _build_tree_dartree_impl(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        greedy: bool = False,
    ) -> WeaverTree:
        """DARTree construction: fixed-width depth-wise expansion, deferred top-B.

        Weaver builds best-first: each round takes the global top-w of the frontier,
        expands them, and the rounds it does not pick are pruned by never being
        expanded. That decision uses only what is known at that round.

        DARTree instead expands level by level with a fixed beam W, accumulates every
        candidate it generates, and prunes ONCE at the end with a global top-B. By
        Lemma 1 that is equivalent to best-first (for beta <= 0, with an ancestor
        tie-break), so it buys best-first quality at wide-batch cost -- which is the
        whole point, since the w-sweep showed round time falls monotonically with
        batch width while tau only degrades past w=4.

        Prefix closure is automatic: score(child) = score(parent) + logprob, and
        logprob <= 0, so a child never outranks its parent and a global top-B can
        never select a node without its ancestors.

        KV is only needed for nodes that get EXPANDED (it is produced by the
        expansion itself and read by descendants' attention), so the cache is sized
        D*W rather than by the full candidate pool.
        """
        bs, depth, pool_size = candidate_ids.shape
        node_budget = int(self.tree_budget)
        num_nodes = node_budget + 1
        device = root_ids.device
        expand_width = min(weaver_tree_expand_width(), int(pool_size))
        beam = max(1, int(envs.SGLANG_DFLASH_TFM_DARTREE_BEAM.get()))
        if not DFlashTfmWorker._dartree_logged:
            DFlashTfmWorker._dartree_logged = True
            # Proof in the run log that this path is live. An env var that never
            # reached the container is otherwise indistinguishable from a null
            # result -- which has already happened once on this workstream.
            logger.info(
                "DFLASH_TFM tree builder: DARTree (beam=%d depth=%d budget=%d "
                "expand_width=%d)",
                beam, depth, node_budget, expand_width,
            )
        beam = min(beam, node_budget)

        num_layers = self.weaver.num_layers
        num_heads = self.weaver.num_heads
        head_dim = self.weaver.d_rank // self.weaver.num_heads

        tokens = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
        parents = torch.full((bs, num_nodes), -1, dtype=torch.long, device=device)
        depths = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
        node_mask = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
        draft_logprobs = torch.full(
            (bs, num_nodes), -torch.inf, dtype=torch.float32, device=device
        )
        is_sampled = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
        tokens[:, 0] = root_ids
        node_mask[:, 0] = True
        draft_logprobs[:, 0] = 0.0
        empty_pool_ids = torch.zeros(
            (bs, num_nodes, pool_size), dtype=torch.long, device=device
        )
        empty_pool_ms = torch.zeros(
            (bs, num_nodes, pool_size), dtype=torch.float32, device=device
        )
        if node_budget <= 0 or depth <= 0:
            return WeaverTree(
                tokens, parents, depths, node_mask, draft_logprobs, is_sampled,
                empty_pool_ids, empty_pool_ms,
            )

        batch_indices = torch.arange(bs, dtype=torch.long, device=device)
        external_keys, external_values, external_mask = (
            self.weaver.prompt_external_kv(output_norm[:, None], proposal_features)
        )
        candidate_ids_rows = candidate_ids.reshape(bs * depth, pool_size)
        candidate_weights_rows = candidate_weights.reshape(
            bs * depth, pool_size, candidate_weights.shape[-1]
        )
        candidate_scores_rows = candidate_scores.reshape(bs * depth, pool_size)

        # KV for expanded nodes only: level d occupies beam slots [d*W, (d+1)*W).
        n_slots = depth * beam + 1
        node_keys = torch.zeros(
            (bs, n_slots, num_layers, num_heads, head_dim),
            dtype=proposal_features.dtype,
            device=device,
        )
        node_values = torch.zeros_like(node_keys)
        slot_ancestors = torch.full(
            (bs, n_slots, depth), -1, dtype=torch.long, device=device
        )
        slot_ancestors[:, 0, 0] = 0

        # Candidate pool: every node ever generated. No KV -- only expanded nodes
        # need that, and pruning is deferred so most of these are never expanded.
        cap = depth * beam * expand_width
        c_tok = torch.zeros((bs, cap), dtype=torch.long, device=device)
        c_par = torch.full((bs, cap), -1, dtype=torch.long, device=device)
        c_dep = torch.zeros((bs, cap), dtype=torch.long, device=device)
        c_score = torch.full((bs, cap), -torch.inf, dtype=torch.float32, device=device)
        c_lp = torch.full((bs, cap), -torch.inf, dtype=torch.float32, device=device)

        # Beam state. cand index -1 marks the root.
        b_cand = torch.full((bs, beam), -1, dtype=torch.long, device=device)
        b_slot = torch.zeros((bs, beam), dtype=torch.long, device=device)
        b_tok = root_ids.to(torch.long)[:, None].expand(bs, beam).contiguous()
        b_dep = torch.zeros((bs, beam), dtype=torch.long, device=device)
        b_score = torch.zeros((bs, beam), dtype=torch.float32, device=device)
        b_active = torch.zeros((bs, beam), dtype=torch.bool, device=device)
        b_active[:, 0] = True                       # level 0 beam is the root alone

        written = 0
        for level in range(depth):
            width = beam
            slot_base = level * beam + 1
            row_batch_indices = (
                batch_indices[:, None].expand(bs, width).reshape(bs * width).contiguous()
            )
            if level == 0:
                # The root has no ancestors; slot_ancestors[:, 0, 0] = 0 is the
                # self-entry descendants read, not an ancestor of the root itself.
                parent_anc = torch.full(
                    (bs, width, depth), -1, dtype=torch.long, device=device
                )
            else:
                parent_anc = torch.gather(
                    slot_ancestors, 1,
                    b_slot.clamp(min=0, max=n_slots - 1)[:, :, None]
                    .expand(bs, width, depth),
                )
            logits, _row_cand_ids, cur_k, cur_v = self._weaver_expand_for_dartree(
                token=b_tok.reshape(bs * width),
                node_depth=b_dep.reshape(bs * width),
                parent_ancestors=parent_anc,
                active=b_active.reshape(bs * width),
                row_batch_indices=row_batch_indices,
                depth=depth,
                candidate_ids_rows=candidate_ids_rows,
                candidate_weights_rows=candidate_weights_rows,
                candidate_scores_rows=candidate_scores_rows,
                external_keys=external_keys,
                external_values=external_values,
                external_mask=external_mask,
                node_keys=node_keys,
                node_values=node_values,
                token_embed=token_embed,
            )
            # This level's expanded nodes take beam slots [slot_base, slot_base+W).
            if level == 0:
                # Root KV lands in slot 0, which every descendant attends to.
                node_keys[:, 0] = cur_k.view(
                    bs, width, num_layers, num_heads, head_dim)[:, 0]
                node_values[:, 0] = cur_v.view(
                    bs, width, num_layers, num_heads, head_dim)[:, 0]
                b_slot = torch.zeros((bs, width), dtype=torch.long, device=device)
            else:
                new_slots = slot_base + torch.arange(width, device=device)
                node_keys[:, new_slots] = cur_k.view(bs, width, num_layers, num_heads, head_dim)
                node_values[:, new_slots] = cur_v.view(bs, width, num_layers, num_heads, head_dim)
                anc = parent_anc.clone()
                d_idx = b_dep.clamp(max=depth - 1)
                anc.scatter_(2, d_idx[:, :, None], b_slot[:, :, None])
                slot_ancestors[:, new_slots] = anc
                b_slot = new_slots[None, :].expand(bs, width).contiguous()

            lg = torch.log_softmax(logits.float(), dim=-1).view(bs, width, pool_size)
            top_lp, top_ix = torch.topk(lg, expand_width, dim=-1)
            child_tok = torch.gather(
                candidate_ids_rows[
                    (row_batch_indices * depth
                     + b_dep.reshape(bs * width).clamp(max=depth - 1))
                ].view(bs, width, pool_size),
                2, top_ix,
            )
            child_score = b_score[:, :, None] + top_lp
            child_ok = b_active[:, :, None] & (child_tok >= 0)
            child_score = torch.where(child_score.isnan(), torch.full_like(child_score, -torch.inf), child_score)
            child_score = child_score.masked_fill(~child_ok, -torch.inf)

            n_new = width * expand_width
            sl = slice(written, written + n_new)
            c_tok[:, sl] = child_tok.reshape(bs, n_new)
            c_dep[:, sl] = (b_dep[:, :, None] + 1).expand(bs, width, expand_width).reshape(bs, n_new)
            c_score[:, sl] = child_score.reshape(bs, n_new)
            c_lp[:, sl] = top_lp.reshape(bs, n_new).masked_fill(
                ~child_ok.reshape(bs, n_new), -torch.inf
            )
            # Parent as a candidate index; -1 means the root.
            c_par[:, sl] = b_cand[:, :, None].expand(bs, width, expand_width).reshape(bs, n_new)
            written += n_new

            if level + 1 >= depth:
                break
            flat_score = child_score.reshape(bs, n_new)
            keep = min(beam, n_new)
            sel_score, sel_ix = torch.topk(flat_score, keep, dim=-1)
            b_cand = (sel_ix + sl.start)
            b_tok = torch.gather(c_tok[:, sl], 1, sel_ix)
            b_dep = torch.gather(c_dep[:, sl], 1, sel_ix)
            b_score = sel_score
            b_active = sel_score > -float("inf")
            if keep < beam:
                pad = beam - keep
                z = lambda t, v: torch.cat([t, torch.full((bs, pad), v, dtype=t.dtype, device=device)], 1)
                b_cand, b_tok, b_dep = z(b_cand, -1), z(b_tok, 0), z(b_dep, 0)
                b_score = z(b_score, -float("inf"))
                b_active = z(b_active, False)

        # Deferred pruning: one global top-B over everything generated.
        take = min(node_budget, written)
        top_score, top_idx = torch.topk(c_score[:, :written], take, dim=-1)
        valid = top_score > -float("inf")
        sel_dep = torch.gather(c_dep[:, :written], 1, top_idx)
        # Order by depth so parents precede children; the verify kernels require it.
        order = torch.argsort(sel_dep + torch.where(valid, 0, depth + 1) * (depth + 1), dim=-1, stable=True)
        top_idx = torch.gather(top_idx, 1, order)
        valid = torch.gather(valid, 1, order)
        sel_dep = torch.gather(sel_dep, 1, order)

        slot_of = torch.full((bs, written), -1, dtype=torch.long, device=device)
        dst = 1 + torch.arange(take, device=device)[None, :].expand(bs, take)
        slot_of.scatter_(1, top_idx, torch.where(valid, dst, torch.full_like(dst, -1)))

        sel_par = torch.gather(c_par[:, :written], 1, top_idx)
        par_slot = torch.where(
            sel_par < 0,
            torch.zeros_like(sel_par),
            torch.gather(slot_of, 1, sel_par.clamp(min=0)),
        )
        keep_mask = valid & (par_slot >= 0)

        tokens[:, 1 : 1 + take] = torch.where(
            keep_mask, torch.gather(c_tok[:, :written], 1, top_idx), torch.zeros_like(top_idx)
        )
        parents[:, 1 : 1 + take] = torch.where(keep_mask, par_slot, torch.full_like(par_slot, -1))
        depths[:, 1 : 1 + take] = torch.where(keep_mask, sel_dep, torch.zeros_like(sel_dep))
        node_mask[:, 1 : 1 + take] = keep_mask
        draft_logprobs[:, 1 : 1 + take] = torch.where(
            keep_mask,
            torch.gather(c_lp[:, :written], 1, top_idx),
            torch.full_like(top_score, -torch.inf),
        )
        # Sample a few trees from REAL traffic. Reporting only the first tree
        # measured a server-warmup tree built on synthetic tokens, which is
        # identical across configurations and told us nothing about the beam.
        DFlashTfmWorker._dartree_calls += 1
        if DFlashTfmWorker._dartree_calls in (200, 800, 2000):
            # Structural report, once. A DARTree tree that silently shrinks looks
            # exactly like a DARTree tree that does not help.
            with torch.no_grad():
                n_active = int(node_mask[:, 1:].sum(1).float().mean().item())
                dropped = int(
                    (valid & (par_slot < 0)).sum(1).float().mean().item()
                )
                dmax = int(depths.max().item())
                dmean = float(
                    depths[:, 1:][node_mask[:, 1:]].float().mean().item()
                ) if bool(node_mask[:, 1:].any()) else 0.0
                # parent[i] < i is required by every downstream verify kernel
                idx = torch.arange(num_nodes, device=device)[None, :].expand_as(parents)
                topo_bad = int(
                    ((parents >= idx) & node_mask).sum().item()
                )
                # prefix closure: an active node's parent must be active
                pmask = torch.gather(
                    node_mask, 1, parents.clamp(min=0)
                )
                closure_bad = int(
                    (node_mask[:, 1:] & ~pmask[:, 1:]).sum().item()
                )
            logger.info(
                "DFLASH_TFM DARTree tree[%d]: active=%d/%d dropped_no_parent=%d "
                "depth_max=%d depth_mean=%.2f topo_violations=%d "
                "closure_violations=%d",
                DFlashTfmWorker._dartree_calls, n_active, node_budget, dropped,
                dmax, dmean, topo_bad, closure_bad,
            )
        return WeaverTree(
            tokens, parents, depths, node_mask, draft_logprobs, is_sampled,
            empty_pool_ids, empty_pool_ms,
        )

    def _build_tree_impl(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        greedy: bool = False,
    ) -> WeaverTree:
        bs, depth, pool_size = candidate_ids.shape
        node_budget = int(self.tree_budget)
        num_nodes = node_budget + 1
        device = root_ids.device
        node_indices = torch.arange(num_nodes, dtype=torch.long, device=device)
        tokens = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
        parents = torch.full((bs, num_nodes), -1, dtype=torch.long, device=device)
        depths = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
        node_mask = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
        draft_logprobs = torch.full(
            (bs, num_nodes), -torch.inf, dtype=torch.float32, device=device
        )
        is_sampled = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
        tokens[:, 0] = root_ids
        node_mask[:, 0] = True
        draft_logprobs[:, 0] = 0.0
        if node_budget <= 0 or depth <= 0:
            return WeaverTree(
                tokens, parents, depths, node_mask, draft_logprobs, is_sampled,
                torch.zeros((bs, num_nodes, pool_size), dtype=torch.long,
                            device=device),
                torch.zeros((bs, num_nodes, pool_size), dtype=torch.float32,
                            device=device),
            )

        expand_width = min(weaver_tree_expand_width(), int(pool_size))
        frontier_slots = (node_budget + 1) * expand_width
        batch_indices = torch.arange(bs, dtype=torch.long, device=device)
        num_layers = self.weaver.num_layers
        num_heads = self.weaver.num_heads
        head_dim = self.weaver.d_rank // self.weaver.num_heads
        batch_expand_width = min(
            weaver_tree_batch_expand_width(node_budget), node_budget
        )

        external_keys, external_values, external_mask = (
            self.weaver.prompt_external_kv(output_norm[:, None], proposal_features)
        )
        candidate_ids_rows = candidate_ids.reshape(bs * depth, pool_size)
        candidate_weights_rows = candidate_weights.reshape(
            bs * depth, pool_size, candidate_weights.shape[-1]
        )
        candidate_scores_rows = candidate_scores.reshape(bs * depth, pool_size)
        node_keys = torch.zeros(
            (bs, num_nodes, num_layers, num_heads, head_dim),
            dtype=proposal_features.dtype,
            device=device,
        )
        node_values = torch.zeros_like(node_keys)
        slot_ancestors = torch.full(
            (bs, num_nodes, depth), -1, dtype=torch.long, device=device
        )
        slot_ancestors[:, 0, 0] = 0

        frontier_tokens = torch.zeros((bs, frontier_slots), dtype=torch.long, device=device)
        frontier_parents = torch.zeros(
            (bs, frontier_slots), dtype=torch.long, device=device
        )
        frontier_depths = torch.zeros(
            (bs, frontier_slots), dtype=torch.long, device=device
        )
        frontier_scores = torch.full(
            (bs, frontier_slots), -torch.inf, dtype=torch.float32, device=device
        )
        frontier_logprobs = torch.full_like(frontier_scores, -torch.inf)
        frontier_active = torch.zeros(
            (bs, frontier_slots), dtype=torch.bool, device=device
        )
        frontier_is_sampled = torch.zeros(
            (bs, frontier_slots), dtype=torch.bool, device=device
        )
        # Preallocated so the residual draw stays CUDA-graph capturable: uniform_()
        # fills in place, whereas torch.rand inside the loop would allocate on replay.
        frontier_uniforms = torch.empty(
            (bs, frontier_slots), dtype=torch.float32, device=device
        )
        frontier_uniforms.uniform_()
        node_pool_ids = torch.zeros(
            (bs, num_nodes, pool_size), dtype=torch.long, device=device
        )
        node_pool_ms = torch.zeros(
            (bs, num_nodes, pool_size), dtype=torch.float32, device=device
        )
        if device.type != "cuda":
            raise RuntimeError("Weaver tree construction requires Triton on CUDA.")
        if expand_width < 2 or expand_width & (expand_width - 1):
            # The frontier kernel is written generically in EXPAND_WIDTH
            # (tl.static_range, child_base * EXPAND_WIDTH), but Triton reductions
            # want a power of two, and only 8 has ever been exercised.
            raise ValueError(
                f"expand_width must be a power of two >= 2, got {expand_width}."
            )

        def write_candidate_frontier(
            logits: torch.Tensor,
            row_candidate_ids: torch.Tensor,
            prefix_score: torch.Tensor,
            node_depth: torch.Tensor,
            active: torch.Tensor,
            slot_start: int,
            width: int,
        ) -> None:
            block_pool = triton.next_power_of_2(int(pool_size))
            _weaver_candidate_frontier_kernel[(logits.shape[0],)](
                logits,
                row_candidate_ids,
                prefix_score,
                node_depth,
                active,
                frontier_tokens,
                frontier_parents,
                frontier_depths,
                frontier_scores,
                frontier_logprobs,
                frontier_active,
                frontier_is_sampled,
                frontier_uniforms,
                node_pool_ids,
                node_pool_ms,
                int(slot_start),
                WIDTH=int(width),
                POOL_SIZE=int(pool_size),
                EXPAND_WIDTH=int(expand_width),
                DEPTH=int(depth),
                FRONTIER_SLOTS=int(frontier_slots),
                BLOCK_POOL=int(block_pool),
                NUM_NODES=int(num_nodes),
                GREEDY=int(bool(greedy)),
            )

        def write_current_slot_cache(
            current_keys: torch.Tensor,
            current_values: torch.Tensor,
            parent_ancestors: torch.Tensor,
            valid: torch.Tensor,
            node_depth: torch.Tensor,
            slot_start: int,
            width: int,
        ) -> None:
            total_kv = bs * width * num_layers * num_heads * head_dim
            total_ancestors = bs * width * depth
            block_size = 256
            grid = (triton.cdiv(max(total_kv, total_ancestors), block_size),)
            _weaver_current_cache_write_kernel[grid](
                current_keys,
                current_values,
                node_keys,
                node_values,
                parent_ancestors,
                slot_ancestors,
                valid,
                node_depth,
                int(slot_start),
                BS=int(bs),
                WIDTH=int(width),
                DEPTH=int(depth),
                NUM_NODES=int(num_nodes),
                NUM_LAYERS=int(num_layers),
                NUM_HEADS=int(num_heads),
                HEAD_DIM=int(head_dim),
                TOTAL_KV=int(total_kv),
                TOTAL_ANCESTORS=int(total_ancestors),
                BLOCK_SIZE=int(block_size),
            )

        def expand_node_indexed(
            token: torch.Tensor,
            node_depth: torch.Tensor,
            parent_ancestors: torch.Tensor,
            active: torch.Tensor,
            row_batch_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            depth_index = node_depth.clamp(max=depth - 1)
            candidate_row_index = row_batch_indices * depth + depth_index
            row_candidate_ids = candidate_ids_rows[candidate_row_index]
            step_kwargs = dict(
                token_ids=torch.where(active, token, torch.zeros_like(token)),
                candidate_ids=row_candidate_ids,
                candidate_weights=candidate_weights_rows[candidate_row_index],
                candidate_scores=candidate_scores_rows[candidate_row_index],
                external_keys=external_keys,
                external_values=external_values,
                external_mask=external_mask,
                position_ids=depth_index,
                node_keys=node_keys,
                node_values=node_values,
                parent_ancestors=parent_ancestors.reshape(
                    bs * parent_ancestors.shape[1], depth
                ).contiguous(),
                row_batch_indices=row_batch_indices,
                token_embed=token_embed,
            )
            logits, current_keys, current_values = (
                self._weaver_indexed_step_compiled(
                    **step_kwargs,
                )
            )
            return logits.float(), row_candidate_ids, current_keys, current_values

        root_parent_ancestors = torch.full(
            (bs, 1, depth), -1, dtype=torch.long, device=device
        )
        root_prefix_score = torch.zeros((bs,), dtype=torch.float32, device=device)
        root_depth = torch.zeros((bs,), dtype=torch.long, device=device)
        root_active = torch.ones((bs,), dtype=torch.bool, device=device)
        root_logits, root_candidate_ids, root_keys, root_values = expand_node_indexed(
            root_ids,
            root_depth,
            root_parent_ancestors,
            root_active,
            batch_indices,
        )
        write_current_slot_cache(
            root_keys,
            root_values,
            root_parent_ancestors,
            root_active[:, None],
            root_depth[:, None],
            0,
            1,
        )
        write_candidate_frontier(
            root_logits,
            root_candidate_ids,
            root_prefix_score,
            root_depth,
            root_active,
            0,
            1,
        )

        def gather_parent_ancestors(parent: torch.Tensor) -> torch.Tensor:
            width = parent.shape[1]
            gather_index = parent.clamp(min=0, max=num_nodes - 1)[:, :, None].expand(
                bs, width, depth
            )
            return torch.gather(slot_ancestors, 1, gather_index)

        # Read once: this is the per-node hot loop.
        depth_bonus = envs.SGLANG_DFLASH_TFM_DEPTH_BONUS.get()
        if not DFlashTfmWorker._knobs_logged:
            DFlashTfmWorker._knobs_logged = True
            # Emitted once per process so a run log carries proof of which knob
            # values actually took effect — an env var that silently failed to
            # reach the container is otherwise indistinguishable from a null result.
            logger.info(
                "DFLASH_TFM tree knobs: expand_unit=%d depth_bonus=%g batch_expand_width=%d",
                envs.SGLANG_DFLASH_TFM_EXPAND_UNIT.get(),
                depth_bonus,
                batch_expand_width,
            )
        if depth_bonus > 0.0:
            raise ValueError(
                "SGLANG_DFLASH_TFM_DEPTH_BONUS must be <= 0; a positive depth "
                "bonus breaks prefix-closure of the selected tree."
            )

        row_base = batch_indices[:, None]
        slot_start = 1
        while slot_start <= node_budget:
            width = min(batch_expand_width, node_budget - slot_start + 1)
            slot_stop = slot_start + width
            slot_slice = slice(slot_start, slot_stop)
            slot_indices = node_indices[slot_slice]
            priorities = frontier_scores
            if depth_bonus:
                # DARTree s_beta: prefix score plus a (non-positive) depth bonus.
                priorities = priorities + depth_bonus * frontier_depths.to(
                    priorities.dtype
                )
            masked_priorities = priorities.masked_fill(~frontier_active, -torch.inf)
            _, frontier_index = torch.topk(masked_priorities, width, dim=1)
            valid = frontier_active.gather(1, frontier_index)
            token = frontier_tokens.gather(1, frontier_index)
            parent = frontier_parents.gather(1, frontier_index)
            node_depth = frontier_depths.gather(1, frontier_index)
            node_score = frontier_scores.gather(1, frontier_index)
            node_logprob = frontier_logprobs.gather(1, frontier_index)
            node_sampled = frontier_is_sampled.gather(1, frontier_index)

            tokens[:, slot_slice] = torch.where(
                valid, token, torch.zeros_like(token)
            )
            parents[:, slot_slice] = torch.where(
                valid, parent, torch.full_like(parent, -1)
            )
            depths[:, slot_slice] = torch.where(
                valid, node_depth, torch.zeros_like(node_depth)
            )
            node_mask[:, slot_slice] = valid
            if _dcut_probe_active():
                _dcut_record(node_score, valid)
            draft_logprobs[:, slot_slice] = torch.where(
                valid, node_logprob, torch.full_like(node_logprob, -torch.inf)
            )
            is_sampled[:, slot_slice] = valid & node_sampled
            frontier_active.scatter_(1, frontier_index, False)

            if slot_stop > node_budget:
                break

            parent_ancestors = gather_parent_ancestors(parent)
            token_flat = token.reshape(bs * width)
            node_score_flat = node_score.reshape(bs * width)
            node_depth_flat = node_depth.reshape(bs * width)
            valid_flat = valid.reshape(bs * width)
            row_batch_indices = (
                row_base.expand(bs, width).reshape(bs * width).contiguous()
            )

            logits, row_candidate_ids, current_keys, current_values = expand_node_indexed(
                token_flat,
                node_depth_flat,
                parent_ancestors,
                valid_flat,
                row_batch_indices,
            )
            write_current_slot_cache(
                current_keys,
                current_values,
                parent_ancestors,
                valid,
                node_depth,
                slot_start,
                width,
            )
            write_candidate_frontier(
                logits,
                row_candidate_ids,
                node_score_flat,
                node_depth_flat,
                valid_flat,
                slot_start,
                width,
            )
            slot_start = slot_stop
        if _dcut_probe_active():
            _DCUT_STATE["calls"] += 1
            if _DCUT_STATE["calls"] in (200, 800, 2000):
                _dcut_report(bs, node_budget)
            else:
                _DCUT_STATE["scores"] = []
        return WeaverTree(
            tokens, parents, depths, node_mask, draft_logprobs, is_sampled,
            node_pool_ids, node_pool_ms,
        )

    def _build_chain_impl(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
        proposal_uniforms: Optional[torch.Tensor] = None,
        draft_tokens_out: Optional[torch.Tensor] = None,
        proposal_tokens_out: Optional[torch.Tensor] = None,
        proposal_probs_out: Optional[torch.Tensor] = None,
    ) -> WeaverChain:
        bs, candidate_depth, pool_size = candidate_ids.shape
        draft_token_num = int(draft_token_num)
        chain_depth = draft_token_num - 1
        block_size = int(self.block_size)
        if candidate_depth + 1 != block_size:
            raise RuntimeError(
                "DFLASH_TFM chain requires candidate depth to match "
                f"block_size - 1, got depth={candidate_depth}, block_size={block_size}."
            )
        if draft_token_num < 1 or draft_token_num > block_size:
            raise RuntimeError(
                "DFLASH_TFM chain draft_token_num must be in [1, block_size], "
                f"got draft_token_num={draft_token_num}, block_size={block_size}."
            )
        device = root_ids.device
        batch_indices = torch.arange(bs, dtype=torch.long, device=device)
        if draft_tokens_out is None:
            draft_tokens = torch.empty(
                (bs, draft_token_num), dtype=torch.long, device=device
            )
        else:
            draft_tokens = draft_tokens_out
        draft_tokens[:, 0] = root_ids.to(torch.long)
        num_layers = self.weaver.num_layers
        num_heads = self.weaver.num_heads
        head_dim = self.weaver.d_rank // self.weaver.num_heads

        external_keys, external_values, external_mask = (
            self.weaver.prompt_external_kv(
                output_norm[:, None],
                proposal_features,
            )
        )
        candidate_ids_rows = candidate_ids.reshape(bs * candidate_depth, pool_size)
        candidate_weights_rows = candidate_weights.reshape(
            bs * candidate_depth, pool_size, candidate_weights.shape[-1]
        )
        candidate_scores_rows = candidate_scores.reshape(bs * candidate_depth, pool_size)
        chain_keys = torch.empty(
            (bs, chain_depth, num_layers, num_heads, head_dim),
            dtype=proposal_features.dtype,
            device=device,
        )
        chain_values = torch.empty_like(chain_keys)
        token = draft_tokens[:, 0]
        do_sample = sampling_info is not None and not sampling_info.is_all_greedy
        if do_sample:
            if proposal_tokens_out is None:
                proposal_tokens = torch.empty(
                    (bs, chain_depth, pool_size), dtype=torch.long, device=device
                )
            else:
                proposal_tokens = proposal_tokens_out
            if proposal_probs_out is None:
                proposal_probs = torch.empty(
                    (bs, chain_depth, pool_size), dtype=torch.float32, device=device
                )
            else:
                proposal_probs = proposal_probs_out
            proposal_tokens.fill_(-1)
            proposal_probs.zero_()
            if proposal_uniforms is None:
                proposal_uniforms = torch.rand(
                    (chain_depth, bs), dtype=torch.float32, device=device
                )
            elif proposal_uniforms.shape != (chain_depth, bs):
                raise ValueError(
                    "proposal_uniforms shape mismatch for DFLASH_TFM chain, "
                    f"got {tuple(proposal_uniforms.shape)}, expected {(chain_depth, bs)}."
                )
        else:
            proposal_tokens = None
            proposal_probs = None
            proposal_uniforms = None

        for step in range(chain_depth):
            row_index = batch_indices * candidate_depth + step
            token_position_ids = torch.full(
                (bs,), step, dtype=torch.long, device=device
            )
            logits, current_keys, current_values = self._weaver_chain_step_compiled(
                token_ids=token,
                candidate_ids=candidate_ids_rows[row_index],
                candidate_weights=candidate_weights_rows[row_index],
                candidate_scores=candidate_scores_rows[row_index],
                external_keys=external_keys,
                external_values=external_values,
                external_mask=external_mask,
                position_ids=token_position_ids,
                chain_keys=chain_keys,
                chain_values=chain_values,
                token_embed=token_embed,
            )
            chain_keys[:, step].copy_(current_keys.permute(1, 0, 2, 3))
            chain_values[:, step].copy_(current_values.permute(1, 0, 2, 3))
            if do_sample:
                token, step_proposal_tokens, step_proposal_probs = (
                    sample_dflash_proposal_from_logits(
                        logits=logits,
                        sampling_info=sampling_info,
                        steps_per_batch=1,
                        token_ids=candidate_ids_rows[row_index],
                        uniform_samples=proposal_uniforms[step],
                    )
                )
                support_width = step_proposal_tokens.shape[1]
                proposal_tokens[:, step, :support_width].copy_(step_proposal_tokens)
                proposal_probs[:, step, :support_width].copy_(step_proposal_probs)
            else:
                next_index = torch.argmax(logits, dim=-1)
                token = candidate_ids_rows[row_index].gather(
                    1, next_index[:, None]
                ).squeeze(1)
            draft_tokens[:, step + 1] = token

        return WeaverChain(draft_tokens, proposal_tokens, proposal_probs)

    def _capture_weaver_chain_cuda_graph(
        self,
        *,
        key: tuple[object, ...],
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
    ) -> WeaverChainCudaGraph:
        device = root_ids.device
        if device.type != "cuda":
            raise RuntimeError("_build_chain CUDA Graph requires a CUDA device.")
        do_sample = sampling_info is not None and not sampling_info.is_all_greedy
        bs, candidate_depth, pool_size = candidate_ids.shape
        chain_depth = int(draft_token_num) - 1
        root_ids_buffer = torch.empty_like(root_ids)
        output_norm_buffer = torch.empty_like(output_norm)
        candidate_ids_buffer = torch.empty_like(candidate_ids)
        candidate_weights_buffer = torch.empty_like(candidate_weights)
        candidate_scores_buffer = torch.empty_like(candidate_scores)
        proposal_features_buffer = torch.empty_like(proposal_features)
        draft_tokens_buffer = torch.empty(
            (bs, int(draft_token_num)), dtype=torch.long, device=device
        )
        proposal_tokens_buffer = (
            torch.empty((bs, chain_depth, pool_size), dtype=torch.long, device=device)
            if do_sample
            else None
        )
        proposal_probs_buffer = (
            torch.empty((bs, chain_depth, pool_size), dtype=torch.float32, device=device)
            if do_sample
            else None
        )
        graph_sampling_info = None
        if do_sample:
            graph_sampling_info = WeaverChainGraphSamplingInfo(
                temperatures=torch.empty_like(sampling_info.temperatures),
                top_ps=torch.empty_like(sampling_info.top_ps),
                top_ks=torch.empty_like(sampling_info.top_ks),
                is_all_greedy=False,
                need_top_p_sampling=bool(
                    getattr(sampling_info, "need_top_p_sampling", False)
                ),
                need_top_k_sampling=bool(
                    getattr(sampling_info, "need_top_k_sampling", True)
                ),
            )
        proposal_uniforms_buffer = (
            torch.empty((chain_depth, bs), dtype=torch.float32, device=device)
            if do_sample
            else None
        )
        root_ids_buffer.copy_(root_ids)
        output_norm_buffer.copy_(output_norm)
        candidate_ids_buffer.copy_(candidate_ids)
        candidate_weights_buffer.copy_(candidate_weights)
        candidate_scores_buffer.copy_(candidate_scores)
        proposal_features_buffer.copy_(proposal_features)
        if graph_sampling_info is not None:
            graph_sampling_info.temperatures.copy_(sampling_info.temperatures)
            graph_sampling_info.top_ps.copy_(sampling_info.top_ps)
            graph_sampling_info.top_ks.copy_(sampling_info.top_ks)
        if proposal_uniforms_buffer is not None:
            proposal_uniforms_buffer.uniform_()

        with torch.inference_mode():
            self._build_chain_impl(
                root_ids=root_ids_buffer,
                output_norm=output_norm_buffer,
                candidate_ids=candidate_ids_buffer,
                candidate_weights=candidate_weights_buffer,
                candidate_scores=candidate_scores_buffer,
                proposal_features=proposal_features_buffer,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=graph_sampling_info if do_sample else sampling_info,
                proposal_uniforms=proposal_uniforms_buffer,
                draft_tokens_out=draft_tokens_buffer,
                proposal_tokens_out=proposal_tokens_buffer,
                proposal_probs_out=proposal_probs_buffer,
            )
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain = self._build_chain_impl(
                    root_ids=root_ids_buffer,
                    output_norm=output_norm_buffer,
                    candidate_ids=candidate_ids_buffer,
                    candidate_weights=candidate_weights_buffer,
                    candidate_scores=candidate_scores_buffer,
                    proposal_features=proposal_features_buffer,
                    token_embed=token_embed,
                    draft_token_num=draft_token_num,
                    sampling_info=graph_sampling_info if do_sample else sampling_info,
                    proposal_uniforms=proposal_uniforms_buffer,
                    draft_tokens_out=draft_tokens_buffer,
                    proposal_tokens_out=proposal_tokens_buffer,
                    proposal_probs_out=proposal_probs_buffer,
                )
            torch.cuda.synchronize(device)

        graph_state = WeaverChainCudaGraph(
            graph=graph,
            root_ids=root_ids_buffer,
            output_norm=output_norm_buffer,
            candidate_ids=candidate_ids_buffer,
            candidate_weights=candidate_weights_buffer,
            candidate_scores=candidate_scores_buffer,
            proposal_features=proposal_features_buffer,
            draft_tokens=draft_tokens_buffer,
            proposal_uniforms=proposal_uniforms_buffer,
            proposal_tokens=proposal_tokens_buffer,
            proposal_probs=proposal_probs_buffer,
            sampling_info=graph_sampling_info,
        )
        chain_graphs = getattr(self, "_weaver_chain_cuda_graphs", None)
        if chain_graphs is None:
            chain_graphs = {}
            self._weaver_chain_cuda_graphs = chain_graphs
        chain_graphs[key] = graph_state
        return graph_state

    def _build_chain_with_cuda_graph(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
    ) -> WeaverChain:
        do_sample = sampling_info is not None and not sampling_info.is_all_greedy
        key = (
            int(self.block_size),
            int(draft_token_num),
            bool(do_sample),
            bool(getattr(sampling_info, "need_top_k_sampling", False)) if do_sample else False,
            bool(getattr(sampling_info, "need_top_p_sampling", False)) if do_sample else False,
            root_ids.device.index,
            tuple(root_ids.shape),
            root_ids.dtype,
            tuple(output_norm.shape),
            output_norm.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(proposal_features.shape),
            proposal_features.dtype,
            token_embed.data_ptr(),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        chain_graphs = getattr(self, "_weaver_chain_cuda_graphs", None)
        graph_state = None if chain_graphs is None else chain_graphs.get(key)
        if graph_state is None:
            graph_state = self._capture_weaver_chain_cuda_graph(
                key=key,
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=sampling_info,
            )
        graph_state.root_ids.copy_(root_ids)
        graph_state.output_norm.copy_(output_norm)
        graph_state.candidate_ids.copy_(candidate_ids)
        graph_state.candidate_weights.copy_(candidate_weights)
        graph_state.candidate_scores.copy_(candidate_scores)
        graph_state.proposal_features.copy_(proposal_features)
        if graph_state.sampling_info is not None:
            graph_state.sampling_info.temperatures.copy_(sampling_info.temperatures)
            graph_state.sampling_info.top_ps.copy_(sampling_info.top_ps)
            graph_state.sampling_info.top_ks.copy_(sampling_info.top_ks)
        if graph_state.proposal_uniforms is not None:
            graph_state.proposal_uniforms.uniform_()
        graph_state.graph.replay()
        if graph_state.proposal_probs is not None:
            probs = graph_state.proposal_probs
            with torch.inference_mode():
                probs.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                probs.clamp_(min=0.0)
                probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-20))
        return WeaverChain(
            graph_state.draft_tokens,
            graph_state.proposal_tokens,
            graph_state.proposal_probs,
        )

    def _build_chain(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
    ) -> WeaverChain:
        can_graph_sample = (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and not bool(getattr(sampling_info, "need_top_k_sampling", True))
            and not bool(getattr(sampling_info, "need_top_p_sampling", False))
        )
        if (
            root_ids.device.type == "cuda"
            and (
                sampling_info is None
                or sampling_info.is_all_greedy
                or can_graph_sample
            )
        ):
            return self._build_chain_with_cuda_graph(
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=sampling_info,
            )
        return self._build_chain_impl(
            root_ids=root_ids,
            output_norm=output_norm,
            candidate_ids=candidate_ids,
            candidate_weights=candidate_weights,
            candidate_scores=candidate_scores,
            proposal_features=proposal_features,
            token_embed=token_embed,
            draft_token_num=draft_token_num,
            sampling_info=sampling_info,
        )

    def _capture_weaver_tree_cuda_graph(
        self,
        *,
        key: tuple[object, ...],
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        greedy: bool = False,
    ) -> WeaverTreeCudaGraph:
        device = root_ids.device
        if device.type != "cuda":
            raise RuntimeError("_build_tree CUDA Graph requires a CUDA device.")
        root_ids_buffer = torch.empty_like(root_ids)
        output_norm_buffer = torch.empty_like(output_norm)
        candidate_ids_buffer = torch.empty_like(candidate_ids)
        candidate_weights_buffer = torch.empty_like(candidate_weights)
        candidate_scores_buffer = torch.empty_like(candidate_scores)
        proposal_features_buffer = torch.empty_like(proposal_features)
        root_ids_buffer.copy_(root_ids)
        output_norm_buffer.copy_(output_norm)
        candidate_ids_buffer.copy_(candidate_ids)
        candidate_weights_buffer.copy_(candidate_weights)
        candidate_scores_buffer.copy_(candidate_scores)
        proposal_features_buffer.copy_(proposal_features)

        with torch.inference_mode():
            self._build_tree_impl(
                root_ids=root_ids_buffer,
                output_norm=output_norm_buffer,
                candidate_ids=candidate_ids_buffer,
                candidate_weights=candidate_weights_buffer,
                candidate_scores=candidate_scores_buffer,
                proposal_features=proposal_features_buffer,
                token_embed=token_embed,
                greedy=greedy,
            )
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tree = self._build_tree_impl(
                    root_ids=root_ids_buffer,
                    output_norm=output_norm_buffer,
                    candidate_ids=candidate_ids_buffer,
                    candidate_weights=candidate_weights_buffer,
                    candidate_scores=candidate_scores_buffer,
                    proposal_features=proposal_features_buffer,
                    token_embed=token_embed,
                    greedy=greedy,
                )
            torch.cuda.synchronize(device)

        graph_state = WeaverTreeCudaGraph(
            graph=graph,
            root_ids=root_ids_buffer,
            output_norm=output_norm_buffer,
            candidate_ids=candidate_ids_buffer,
            candidate_weights=candidate_weights_buffer,
            candidate_scores=candidate_scores_buffer,
            proposal_features=proposal_features_buffer,
            tree=tree,
        )
        tree_graphs = getattr(self, "_weaver_tree_cuda_graphs", None)
        if tree_graphs is None:
            tree_graphs = {}
            self._weaver_tree_cuda_graphs = tree_graphs
        tree_graphs[key] = graph_state
        return graph_state

    def _build_tree_with_cuda_graph(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        greedy: bool = False,
    ) -> WeaverTree:
        key = (
            bool(greedy),
            int(self.tree_budget),
            int(weaver_tree_batch_expand_width(self.tree_budget)),
            root_ids.device.index,
            tuple(root_ids.shape),
            root_ids.dtype,
            tuple(output_norm.shape),
            output_norm.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(proposal_features.shape),
            proposal_features.dtype,
            token_embed.data_ptr(),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        tree_graphs = getattr(self, "_weaver_tree_cuda_graphs", None)
        graph_state = None if tree_graphs is None else tree_graphs.get(key)
        if graph_state is None:
            graph_state = self._capture_weaver_tree_cuda_graph(
                key=key,
                greedy=greedy,
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
            )
        graph_state.root_ids.copy_(root_ids)
        graph_state.output_norm.copy_(output_norm)
        graph_state.candidate_ids.copy_(candidate_ids)
        graph_state.candidate_weights.copy_(candidate_weights)
        graph_state.candidate_scores.copy_(candidate_scores)
        graph_state.proposal_features.copy_(proposal_features)
        graph_state.graph.replay()
        return graph_state.tree

    def _build_tree(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        greedy: bool = False,
    ) -> WeaverTree:
        if envs.SGLANG_DFLASH_TFM_DARTREE.get():
            # Not graph-captured yet: DARTree's shapes are static, but the path
            # needs to be correct before it is baked into a graph.
            return self._build_tree_dartree_impl(
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                greedy=greedy,
            )
        # DARTree currently runs eager, so a graph-captured stock path would be
        # compared against it unfairly: tree construction is ~15 sequential Weaver
        # calls plus many small ops per level, which is precisely where launch
        # overhead dominates. This knob runs stock eager for an apples-to-apples
        # cost comparison until DARTree is itself captured.
        if root_ids.device.type == "cuda" and not envs.SGLANG_DFLASH_TFM_NO_TREE_GRAPH.get():
            return self._build_tree_with_cuda_graph(
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                greedy=greedy,
            )
        return self._build_tree_impl(
            root_ids=root_ids,
            output_norm=output_norm,
            candidate_ids=candidate_ids,
            candidate_weights=candidate_weights,
            candidate_scores=candidate_scores,
            proposal_features=proposal_features,
            token_embed=token_embed,
            greedy=greedy,
        )

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashTfmDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        if not isinstance(draft_input, DFlashTfmDraftInput):
            raise RuntimeError(
                "DFLASH_TFM decode requires DFlashTfmDraftInput state."
            )
        if batch.has_grammar:
            raise RuntimeError(
                "DFLASH_TFM does not support grammar constraints in the MVP."
            )
        bs = batch.batch_size()
        embed_module, lm_head = self._target_embedding_and_lm_head()
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_size = int(self.block_size)
        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=batch.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH_TFM Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets,
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=batch.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=batch.device,
                )
                verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=batch.device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        input_embeds_2d = embed_module(block_ids)
        input_embeds = input_embeds_2d.view(-1, input_embeds_2d.shape[-1])
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]

        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(
                draft_prefix_lens.to(device="cpu", dtype=torch.int32)
            )

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=batch.req_to_token_pool.req_to_token,
                req_pool_indices=batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            if draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                draft_seq_lens_sum = (
                    int(batch.seq_lens_sum)
                    if batch.seq_lens_sum is not None
                    else int(batch.seq_lens_cpu.sum())
                )
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH_TFM,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError(
                "DFLASH_TFM draft model returned no hidden states."
            )
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        depth = min(self.block_size - 1, self.weaver.K)
        proposal_features = draft_hidden[:, 1 : 1 + depth].contiguous()
        scores, ids = self._topk_from_lm_head(
            proposal_features.reshape(bs * depth, proposal_features.shape[-1]),
            lm_head,
            self.candidate_pool_size,
        )
        candidate_scores = scores.view(bs, depth, -1)
        candidate_ids = ids.view(bs, depth, -1)
        residual_lm_head = self._weaver_residual_lm_head(lm_head)
        candidate_weights = residual_lm_head[candidate_ids.clamp_min(0)]
        token_embed = self._weaver_token_embed(embed_module)
        if self.use_chain_verify:
            draft_token_num = min(int(self.target_verify_tokens), block_size)
            if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
                if not is_dflash_sampling_verify_available():
                    raise RuntimeError(
                        "DFLASH_TFM chain non-greedy proposal sampling "
                        "requires DFlash sampling verify."
                    )
            chain = self._build_chain(
                root_ids=draft_input.bonus_tokens.to(torch.long),
                output_norm=draft_input.output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=batch.sampling_info,
            )
            draft_tokens = chain.draft_tokens
            verify_positions = positions_2d[:, :draft_token_num].reshape(-1)
            verify_out_cache_loc = verify_out_cache_loc_2d[:, :draft_token_num].reshape(-1)
            verify_input = DFlashVerifyInput(
                draft_token=draft_tokens.reshape(-1),
                positions=verify_positions,
                draft_token_num=draft_token_num,
                custom_mask=None,
                proposal_tokens=chain.proposal_tokens,
                proposal_probs=chain.proposal_probs,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            batch.out_cache_loc = verify_out_cache_loc
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            batch.spec_info = verify_input
            batch.return_hidden_states = False
            return
        # Greedy verification is exactly lossless on a pure top-K tree (q is a
        # point mass, so the draft probability cancels out of the acceptance test),
        # and a residual-sampled child would only cost accepted length there. So
        # the sampled child and UniVer are for stochastic sampling only.
        # Two separate signals. real_greedy decides the VERIFY rule (at T=0 the
        # cascade is already exactly lossless); tree_greedy decides the tree SHAPE.
        # Forcing pure top-K at T>0 keeps UniVer on, where the absence of a sampled
        # child makes every node fall back to Z_v = 1 -- membership testing, which
        # is lossless and optimal for a fixed candidate set. That isolates tree
        # quality from the sampled child, which otherwise move together.
        real_greedy = (
            batch.sampling_info is None or batch.sampling_info.is_all_greedy
        )
        tree_greedy = real_greedy or envs.SGLANG_DFLASH_TFM_PURE_TOPK.get()
        tree = self._build_tree(
            root_ids=draft_input.bonus_tokens.to(torch.long),
            output_norm=draft_input.output_norm,
            candidate_ids=candidate_ids,
            candidate_weights=candidate_weights,
            candidate_scores=candidate_scores,
            proposal_features=proposal_features,
            token_embed=token_embed,
            greedy=tree_greedy,
        )
        mask_seq_lens_cpu = batch.seq_lens_cpu
        if mask_seq_lens_cpu is None:
            mask_seq_lens_cpu = batch.seq_lens.to("cpu", dtype=torch.int32)
        (
            custom_mask,
            verify_positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = build_tree_metadata(
            draft_tokens=tree.draft_tokens,
            parent_indices=tree.parent_indices,
            depths=tree.depths,
            node_mask=tree.node_mask,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=mask_seq_lens_cpu,
            max_depth=depth,
        )
        verify_input = DFlashTfmVerifyInput(
            draft_token=tree.draft_tokens.reshape(-1),
            positions=verify_positions,
            draft_token_num=tree.draft_tokens.shape[1],
            custom_mask=custom_mask,
            mask_seq_lens_cpu=mask_seq_lens_cpu,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            depths=tree.depths,
            parent_indices=tree.parent_indices,
            node_mask=tree.node_mask,
            draft_logprobs=tree.draft_logprobs,
            is_sampled=tree.is_sampled,
            pool_ids=tree.pool_ids,
            pool_ms=tree.pool_ms,
            univer_ok=not real_greedy,
        )
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "DFLASH_TFM does not support return_logprob yet."
            )
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_result = self.target_worker.forward_batch_generation(batch, **kwargs)
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            batch_result.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_result.new_seq_lens)
            split = split_dflash_tfm_hidden(
                logits_output.hidden_states, self.hidden_size
            )
            if batch.extend_lens is None or batch.prefix_lens is None:
                raise RuntimeError(
                    "DFLASH_TFM expected extend_lens / prefix_lens in extend mode."
                )
            if batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH_TFM prefill expected out_cache_loc, but got None."
                )
            device = next_token_ids.device

            def _to_int32_device_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device=device, dtype=torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(batch.extend_lens)
            prefix_lens = _to_int32_device_tensor(batch.prefix_lens)
            positions, _ = compute_position(
                self.model_runner.server_args.attention_backend,
                prefix_lens,
                extend_seq_lens,
                int(sum(batch.extend_lens)),
            )
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=split.target_hidden,
                cache_loc=batch.out_cache_loc,
                positions=positions,
            )
            logits_output.hidden_states = None

            draft_input = DFlashTfmDraftInput(
                bonus_tokens=next_token_ids.to(torch.int64),
                new_seq_lens=batch.seq_lens,
                output_norm=split.output_norm[
                    _last_extend_indices(batch.extend_lens, device)
                ],
                committed_seq_lens_cpu=(
                    batch.seq_lens_cpu.clone()
                    if batch.seq_lens_cpu is not None
                    else None
                ),
            )
            batch.spec_info = draft_input
            batch_result.next_draft_input = draft_input
            batch_result.speculative_num_draft_tokens = int(
                self.server_args.speculative_num_draft_tokens
            )
            batch_result.num_correct_drafts = 0
            return batch_result

        if batch.spec_info is None:
            batch.spec_info = DFlashTfmDraftInput.create_idle_input(
                self.device, int(self.weaver.d_model)
            )
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashTfmDraftInput):
            raise RuntimeError(
                "DFLASH_TFM decode requires DFlashTfmDraftInput state."
            )
        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = DFlashTfmDraftInput.create_idle_input(
                self.device, int(self.weaver.d_model)
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=int(
                    self.server_args.speculative_num_draft_tokens
                ),
                new_seq_lens=next_draft_input.new_seq_lens,
            )

        # `seq_lens` may have been produced on another stream in the spec-v2 path.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = batch.batch_size()
        self._prepare_for_speculative_decoding(batch, draft_input)
        assert batch.forward_mode.is_target_verify()
        verify_input = batch.spec_info
        if isinstance(verify_input, DFlashVerifyInput) and not isinstance(
            verify_input, DFlashTfmVerifyInput
        ):
            need_mamba_verify_commit = hasattr(
                self.target_worker.model_runner.attn_backend,
                "update_mamba_state_after_mtp_verify",
            )
            seq_lens_pre_verify = (
                batch.seq_lens.clone() if need_mamba_verify_commit else None
            )
            seq_lens_cpu_backup = batch.seq_lens_cpu
            seq_lens_sum_backup = batch.seq_lens_sum
            if draft_input.reserved_seq_lens_cpu is not None:
                batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
                batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            try:
                verify_forward_batch, can_run_cuda_graph = (
                    verify_input.prepare_for_verify(batch, self.target_worker)
                )
                batch_result = self.target_worker.forward_batch_generation(
                    batch=None,
                    forward_batch=verify_forward_batch,
                    is_verify=True,
                    skip_attn_backend_init=True,
                    **kwargs,
                )
            finally:
                batch.seq_lens_cpu = seq_lens_cpu_backup
                batch.seq_lens_sum = seq_lens_sum_backup

            logits_output = batch_result.logits_output
            sampling_info = batch.sampling_info
            draft_token_num = int(verify_input.draft_token_num)
            if sampling_info is not None:
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                    draft_token_num=draft_token_num,
                )
            candidates = verify_input.draft_token.view(bs, draft_token_num)
            new_seq_lens = None
            if sampling_info is not None and not sampling_info.is_all_greedy:
                accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                    proposal_tokens=verify_input.proposal_tokens,
                    proposal_probs=verify_input.proposal_probs,
                )
                commit_lens = accept_len.to(torch.int32) + 1
                out_tokens = torch.empty(
                    (bs, draft_token_num), dtype=torch.int64, device=batch.device
                )
                if draft_token_num > 1:
                    out_tokens[:, : draft_token_num - 1].copy_(candidates[:, 1:])
                out_tokens[:, draft_token_num - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )
            else:
                target_predict = torch.argmax(
                    logits_output.next_token_logits, dim=-1
                ).view(bs, draft_token_num)
                accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = accept_len.to(torch.int32) + 1
                out_tokens = torch.empty(
                    (bs, draft_token_num),
                    dtype=torch.int64,
                    device=batch.device,
                )
                if draft_token_num > 1:
                    out_tokens[:, : draft_token_num - 1].copy_(
                        candidates[:, 1:]
                    )
                out_tokens[:, draft_token_num - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )
            if new_seq_lens is None:
                new_seq_lens = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)

            if need_mamba_verify_commit:
                assert seq_lens_pre_verify is not None
                self._update_target_mamba_state_after_verify(
                    batch=batch,
                    seq_lens_pre_verify=seq_lens_pre_verify,
                    commit_lens=commit_lens,
                )
            if on_publish is not None:
                on_publish(new_seq_lens)
            split = split_dflash_tfm_hidden(
                logits_output.hidden_states, self.hidden_size
            )
            cache_loc_2d = batch.out_cache_loc.view(bs, draft_token_num)
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=split.target_hidden.reshape(
                    -1, split.target_hidden.shape[-1]
                ),
                cache_loc=batch.out_cache_loc,
                positions=verify_input.positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
            terminal = (
                torch.arange(bs, device=batch.device, dtype=torch.long)
                * draft_token_num
                + accept_len.to(torch.long)
            )
            next_output_norm = split.output_norm[terminal]
            logits_output.hidden_states = None

            committed_seq_lens_cpu = (
                new_seq_lens.to("cpu", dtype=batch.seq_lens_cpu.dtype)
                if batch.seq_lens_cpu is not None
                else None
            )
            next_draft_input = DFlashTfmDraftInput(
                bonus_tokens=bonus,
                new_seq_lens=new_seq_lens,
                output_norm=next_output_norm,
                committed_seq_lens_cpu=committed_seq_lens_cpu,
            )
            batch.spec_info = next_draft_input
            batch.forward_mode = ForwardMode.DECODE
            num_correct_cpu = [int(x) for x in accept_len.to("cpu").tolist()]
            num_correct_drafts = sum(num_correct_cpu)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=out_tokens.reshape(-1),
                accept_lens=commit_lens,
                next_draft_input=next_draft_input,
                speculative_num_draft_tokens=draft_token_num,
                new_seq_lens=new_seq_lens,
                num_correct_drafts=num_correct_drafts,
                num_correct_drafts_per_req_cpu=num_correct_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
                extra_keep_alive_refs=[verify_forward_batch],
            )

        assert isinstance(verify_input, DFlashTfmVerifyInput)

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
            batch,
            self.target_worker,
            self.page_size,
        )
        batch_result = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
            **kwargs,
        )
        logits_output = batch_result.logits_output
        (
            out_tokens,
            commit_lens,
            next_target_hidden,
            next_target_positions,
            next_output_norm,
            num_correct_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
            hidden_size=self.hidden_size,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
        )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
                accept_leaf_slots=verify_input.accept_leaf_slots,
            )
        new_bonus_tokens = out_tokens[
            torch.arange(bs, device=batch.device),
            commit_lens.to(torch.long) - 1,
        ]
        new_seq_lens = batch.seq_lens.clone()
        if on_publish is not None:
            on_publish(new_seq_lens)
        append_cache_loc_2d = None
        append_commit_lens = None
        if (
            self.page_size > 1
            and int(batch.out_cache_loc.numel())
            == bs * int(verify_input.draft_token_num)
        ):
            append_cache_loc_2d = batch.out_cache_loc.view(
                bs, int(verify_input.draft_token_num)
            )
            append_commit_lens = commit_lens
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=next_target_hidden,
            cache_loc=batch.out_cache_loc,
            positions=next_target_positions,
            cache_loc_2d=append_cache_loc_2d,
            commit_lens=append_commit_lens,
        )

        next_draft_input = DFlashTfmDraftInput(
            bonus_tokens=new_bonus_tokens,
            new_seq_lens=new_seq_lens,
            output_norm=next_output_norm,
            committed_seq_lens_cpu=(
                batch.seq_lens_cpu.clone() if batch.seq_lens_cpu is not None else None
            ),
        )
        batch.spec_info = next_draft_input
        batch.forward_mode = ForwardMode.DECODE
        num_correct_drafts = sum(num_correct_cpu)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(
                getattr(
                    verify_input,
                    "draft_token_num",
                    self.server_args.speculative_num_draft_tokens,
                )
            ),
            new_seq_lens=new_seq_lens,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            extra_keep_alive_refs=[verify_forward_batch],
        )
