# -*- coding: utf-8 -*-
# Block-quadratic (chunk-mode) gated-delta-rule kernels for speculative tree
# verification.
#
# The chain chunk kernels (chunk_fwd.py / wy_fast.py / chunk_o.py) compute,
# with G = inclusive sequence cumsum of the log-decay g:
#
#   A[i, j] = beta_i * exp(G_i - G_j) * (k_i . k_j)        (j < i, strictly lower)
#   T~      = (I + A)^{-1}
#   w       = T~ @ (k * beta * exp(G));  u = T~ @ (v * beta)
#   o_i     = scale * (exp(G_i) * q_i @ h0 + sum_{j <= i} (q_i . k_j) exp(G_i - G_j) (u_j - w_j @ h0))
#
# For a draft tree (tokens topologically ordered, parent index < child index,
# root at 0) the same algebra holds branch-wise after two substitutions:
#
#   sequence cumsum  ->  path cumsum      G_i = g_i + G_parent(i)
#   lower-triangular ->  ancestor mask    A[i, j] != 0 iff j is a proper ancestor of i
#
# Cross-branch entries of (I + A)^{-1} are exactly zero (powers of A only
# connect tokens along ancestor chains), so T~ restricted to any root->leaf
# path equals the chain T~ of that path. A chain tree therefore reproduces
# chunk_gated_delta_rule bit-for-bit up to kernel scheduling.
#
# These kernels compute verification outputs ONLY: no per-token state caching
# and no state write-back. State commit after acceptance is a separate pass
# over the accepted path (see gdn_backend).
#
# Layout is dense [B, T, H, D] with one tree per request, T <= 64 (one chunk).

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.op import exp, safe_exp
from sglang.srt.layers.attention.fla.utils import autotune_cache_kwargs, is_tf32_supported
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd

if is_tf32_supported:
    _MERGE_DOT_PRECISION = tl.constexpr("tf32")
else:
    _MERGE_DOT_PRECISION = tl.constexpr("ieee")

NEUMANN_DEFAULT_TERMS = 4
TREE_CHUNK_SIZE = 64


def tree_mask_capacity(T: int) -> int:
    """Bitset slot capacity for T draft tokens: power of two, at least 64.

    Power-of-two keeps the word count (capacity / 64) a power of two so the
    kernels can tl.arange over words, and keeps the T <= 64 layout
    byte-identical to the original single-word [B, 64] masks.
    """
    cap = TREE_CHUNK_SIZE
    while cap < T:
        cap *= 2
    return cap


@triton.jit(do_not_specialize=["T"])
def tree_ancestor_bitmask_kernel(
    parent_tokens,
    ancestor_masks,
    T,
    stride_parent_seq,
    stride_parent_token,
    BT: tl.constexpr,
    NW: tl.constexpr,
):
    """ancestor_masks[n, i, w] = int64 bitset word w (tokens 64w..64w+63) of
    the proper ancestors of token i.

    parent_tokens follows the retrieve_parent_token convention: tree-local
    parent index per token, token 0 is the root (its parent entry is ignored).
    """
    i_n = tl.program_id(0)
    o_t = tl.arange(0, BT)
    o_w = tl.arange(0, NW)
    m_t = o_t < T

    b_parents = tl.load(
        parent_tokens + i_n * stride_parent_seq + o_t * stride_parent_token,
        mask=m_t,
        other=0,
    ).to(tl.int64)
    # b_bits[i, w] = token i's own bit, one-hot in its word
    b_bit = tl.full([BT], 1, tl.int64) << (o_t % 64).to(tl.int64)
    b_bits = tl.where((o_t // 64)[:, None] == o_w[None, :], b_bit[:, None], 0)
    b_anc = tl.zeros([BT, NW], tl.int64)
    for i in range(1, T):
        parent = tl.sum(tl.where(o_t == i, b_parents, 0))
        anc_parent = tl.sum(tl.where(o_t[:, None] == parent, b_anc, 0), 0)
        bit_parent = tl.sum(tl.where(o_t[:, None] == parent, b_bits, 0), 0)
        b_anc = tl.where(
            o_t[:, None] == i, (anc_parent | bit_parent)[None, :], b_anc
        )
    tl.store(
        ancestor_masks + (i_n * BT + o_t[:, None]) * NW + o_w[None, :],
        b_anc,
        mask=m_t[:, None],
    )


def build_tree_ancestor_masks(
    parent_tokens: torch.Tensor,
    T: int,
    BT: int = None,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Build [B, BT, BT // 64] int64 proper-ancestor bitsets from tree-local
    parent indices. Word w of row i covers ancestors with index in
    [64w, 64w + 64); for BT == 64 the layout is byte-identical to the original
    single-word [B, 64] masks.

    Layer-invariant: build once per verify step and reuse across all GDN layers.
    Pass a preallocated `out` to stay allocation-free under CUDA graph capture.
    """
    B = parent_tokens.shape[0]
    if out is not None:
        ancestor_masks = out[:B]
        assert ancestor_masks.dim() == 3 and ancestor_masks.dtype == torch.int64
        BT, NW = ancestor_masks.shape[-2], ancestor_masks.shape[-1]
    else:
        if BT is None:
            BT = tree_mask_capacity(T)
        NW = BT // 64
        ancestor_masks = torch.zeros(
            B, BT, NW, dtype=torch.int64, device=parent_tokens.device
        )
    assert BT & (BT - 1) == 0 and NW == BT // 64, (BT, NW)
    assert T <= BT, f"tree verify supports at most {BT} draft tokens, got {T}"
    tree_ancestor_bitmask_kernel[(B,)](
        parent_tokens=parent_tokens,
        ancestor_masks=ancestor_masks,
        T=T,
        stride_parent_seq=parent_tokens.stride(0),
        stride_parent_token=parent_tokens.stride(1),
        BT=BT,
        NW=NW,
    )
    return ancestor_masks


@triton.jit(do_not_specialize=["T"])
def tree_parent_tokens_kernel(
    next_tokens,
    next_siblings,
    parent_tokens,
    T,
    stride_next_seq,
    stride_next_token,
    stride_sibling_seq,
    stride_sibling_token,
    stride_parent_seq,
    stride_parent_token,
    BT: tl.constexpr,
):
    """Derive tree-local parent indices from retrieve_next_token /
    retrieve_next_sibling, once per verify step.

    Same derivation the tree-aware conv kernel performs inline, lifted out so
    it runs once instead of per layer per channel block.
    """
    i_n = tl.program_id(0)
    o_t = tl.arange(0, BT)
    m_t = o_t < T

    b_next = tl.load(
        next_tokens + i_n * stride_next_seq + o_t * stride_next_token,
        mask=m_t,
        other=-1,
    )
    b_sibling = tl.load(
        next_siblings + i_n * stride_sibling_seq + o_t * stride_sibling_token,
        mask=m_t,
        other=-1,
    )
    b_parents = tl.zeros([BT], dtype=tl.int32)
    for i in range(0, T):
        child = tl.sum(tl.where(o_t == i, b_next, 0))
        if child != -1:
            b_parents = tl.where(o_t == child, i, b_parents)
        sibling = tl.sum(tl.where(o_t == i, b_sibling, 0))
        if sibling != -1:
            parent = tl.sum(tl.where(o_t == i, b_parents, 0))
            b_parents = tl.where(o_t == sibling, parent, b_parents)
    tl.store(
        parent_tokens + i_n * stride_parent_seq + o_t * stride_parent_token,
        b_parents,
        mask=m_t,
    )


def derive_tree_parent_tokens(
    next_tokens: torch.Tensor,
    next_siblings: torch.Tensor,
    parent_tokens: torch.Tensor,
    T: int,
):
    """Fill parent_tokens[b, i] with token i's tree-local parent (root: 0)."""
    B = next_tokens.shape[0]
    tree_parent_tokens_kernel[(B,)](
        next_tokens=next_tokens,
        next_siblings=next_siblings,
        parent_tokens=parent_tokens,
        T=T,
        stride_next_seq=next_tokens.stride(0),
        stride_next_token=next_tokens.stride(1),
        stride_sibling_seq=next_siblings.stride(0),
        stride_sibling_token=next_siblings.stride(1),
        stride_parent_seq=parent_tokens.stride(0),
        stride_parent_token=parent_tokens.stride(1),
        BT=triton.next_power_of_2(max(T, 16)),
    )


@triton.jit
def tree_verify_conv1d_kernel(
    x,
    weight,
    bias,
    conv_states,
    conv_state_indices,
    parent_tokens,
    intermediate_conv_window,
    intermediate_state_indices,
    o,
    T: tl.constexpr,
    dim: tl.constexpr,
    state_len: tl.constexpr,
    stride_x_seq,
    stride_x_dim,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    stride_parent_seq,
    stride_parent_token,
    stride_win_seq,
    stride_win_step,
    stride_win_dim,
    stride_win_tok,
    stride_o_seq,
    stride_o_token,
    stride_o_dim,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tree conv for target verify, parallel over (request, token, channels).

    Token i convolves over its ancestor path; entries above the root come from
    the committed conv state (most recent at column state_len-1 ... oldest at
    state_len-KERNEL_WIDTH+1). Writes per-token conv windows for the
    post-accept commit, never the committed state itself.

    Replaces _causal_conv1d_update_kernel's tree branch, which is sequential
    over tokens and re-derives parents per layer per channel block.
    """
    i_bt, i_nb = tl.program_id(0), tl.program_id(1)
    i_b, i_t = i_bt // T, i_bt % T

    slot = tl.load(conv_state_indices + i_b).to(tl.int64)
    if slot == pad_slot_id:
        return

    o_d = i_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    m_d = o_d < dim

    p_x = x + i_b * stride_x_seq + o_d * stride_x_dim
    p_cs = conv_states + slot * stride_cs_seq + o_d * stride_cs_dim
    p_parent = parent_tokens + i_b * stride_parent_seq

    if HAS_BIAS:
        b_acc = tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)
    else:
        b_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    if SAVE_INTERMEDIATE:
        win_idx = tl.load(intermediate_state_indices + i_b).to(tl.int64)
        p_win = (
            intermediate_conv_window
            + win_idx * stride_win_seq
            + i_t * stride_win_step
            + o_d * stride_win_dim
        )

    # Walk the ancestor chain: cur >= 0 is a tree slot, cur < 0 selects the
    # committed state column state_len + cur.
    cur = i_t
    for j in tl.static_range(KERNEL_WIDTH):
        from_tree = cur >= 0
        b_x = tl.load(
            p_x + tl.maximum(cur, 0).to(tl.int64) * stride_x_token,
            mask=m_d & from_tree,
            other=0.0,
        )
        col = tl.maximum(state_len + tl.minimum(cur, -1), 0).to(tl.int64)
        b_cs = tl.load(
            p_cs + col * stride_cs_tok,
            mask=m_d & (cur < 0),
            other=0.0,
        )
        b_val = tl.where(from_tree, b_x, b_cs)

        b_w = tl.load(
            weight + o_d * stride_w_dim + (KERNEL_WIDTH - 1 - j) * stride_w_width,
            mask=m_d,
            other=0.0,
        )
        b_acc += b_val.to(tl.float32) * b_w.to(tl.float32)

        if SAVE_INTERMEDIATE:
            if KERNEL_WIDTH - j - 2 >= 0:
                tl.store(
                    p_win + (KERNEL_WIDTH - j - 2) * stride_win_tok,
                    b_val,
                    mask=m_d,
                )

        if cur > 0:
            cur = tl.load(p_parent + cur.to(tl.int64) * stride_parent_token).to(
                tl.int32
            )
        else:
            cur = cur - 1

    if SILU_ACTIVATION:
        b_acc = b_acc / (1 + tl.exp(-b_acc))
    tl.store(
        o + i_b * stride_o_seq + i_t * stride_o_token + o_d * stride_o_dim,
        b_acc.to(o.dtype.element_ty),
        mask=m_d,
    )


def tree_verify_conv1d_update(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    conv_state_indices: torch.Tensor,
    parent_tokens: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Tree conv over draft tokens for target verify.

    x: [B, dim, T] (any strides), conv_states: [pool, dim, state_len],
    weight: [dim, width]. Returns [B, dim, T] (a transposed view of a
    contiguous [B, T, dim] buffer). The committed conv state is read-only.
    """
    B, dim, T = x.shape
    width = weight.shape[1]
    assert width == 4, f"GDN conv width 4 expected, got {width}"
    assert activation in (None, "silu", "swish")
    assert intermediate_conv_window is not None

    out = x.new_empty(B, T, dim)
    grid = (B * T, triton.cdiv(dim, 512))
    tree_verify_conv1d_kernel[grid](
        x=x,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        conv_state_indices=conv_state_indices,
        parent_tokens=parent_tokens,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        o=out,
        T=T,
        dim=dim,
        state_len=conv_states.shape[-1],
        stride_x_seq=x.stride(0),
        stride_x_dim=x.stride(1),
        stride_x_token=x.stride(2),
        stride_w_dim=weight.stride(0),
        stride_w_width=weight.stride(1),
        stride_cs_seq=conv_states.stride(0),
        stride_cs_dim=conv_states.stride(1),
        stride_cs_tok=conv_states.stride(2),
        stride_parent_seq=parent_tokens.stride(0),
        stride_parent_token=parent_tokens.stride(1),
        stride_win_seq=intermediate_conv_window.stride(0),
        stride_win_step=intermediate_conv_window.stride(1),
        stride_win_dim=intermediate_conv_window.stride(2),
        stride_win_tok=intermediate_conv_window.stride(3),
        stride_o_seq=out.stride(0),
        stride_o_token=out.stride(1),
        stride_o_dim=out.stride(2),
        pad_slot_id=-1,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        BLOCK_N=512,
    )
    return out.transpose(1, 2)


@triton.jit(do_not_specialize=["T"])
def tree_path_cumsum_kernel(
    g,
    g_path,
    ancestor_masks,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    """g_path[n, i, h] = sum of g over token i's root->i path (inclusive)."""
    i_n = tl.program_id(0)
    o_t = tl.arange(0, BT)
    o_h = tl.arange(0, BH)
    m_t = o_t < T
    m_h = o_h < H
    m_th = m_t[:, None] & m_h[None, :]

    b_anc = tl.load(ancestor_masks + i_n * BT + o_t, mask=m_t, other=0)
    m_path = ((b_anc[:, None] >> o_t[None, :].to(tl.int64)) & 1) != 0
    m_path |= o_t[:, None] == o_t[None, :]

    p_g = g + (i_n * T + o_t[:, None]) * H + o_h[None, :]
    b_g = tl.load(p_g, mask=m_th, other=0.0).to(tl.float32)
    b_g_path = tl.dot(m_path.to(tl.float32), b_g, input_precision="ieee")
    tl.store(g_path + (i_n * T + o_t[:, None]) * H + o_h[None, :], b_g_path, mask=m_th)


@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "Hg", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def tree_kkt_solve_kernel(
    k,
    g,
    beta,
    ancestor_masks,
    A,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """Tree-masked variant of chunk_gated_delta_rule_fwd_kkt_solve_kernel.

    Computes (I + A)^{-1} where A[i, j] = beta_i * exp(G_i - G_j) * (k_i . k_j)
    masked to proper-ancestor pairs (the chain kernel's strict lower triangle).
    g must be the tree path cumsum. One chunk per request: T <= BT.

    The forward substitution and block merge are pure triangular algebra and
    are taken verbatim from the chain kernel; only the masking differs.
    """
    i_bh = tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H

    bos = i_b * T

    i_tc0 = 0
    i_tc1 = BC
    i_tc2 = 2 * BC
    i_tc3 = 3 * BC

    k += (bos * Hg + i_h // (H // Hg)) * K
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    p_b0 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    p_g0 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_g1 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_g2 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_g3 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
    b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
    b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
    b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)

    # proper-ancestor bitsets per row sub-chunk
    b_anc0 = tl.load(ancestor_masks + i_b * BT + i_tc0 + o_i, mask=m_tc0, other=0)
    b_anc1 = tl.load(ancestor_masks + i_b * BT + i_tc1 + o_i, mask=m_tc1, other=0)
    b_anc2 = tl.load(ancestor_masks + i_b * BT + i_tc2 + o_i, mask=m_tc2, other=0)
    b_anc3 = tl.load(ancestor_masks + i_b * BT + i_tc3 + o_i, mask=m_tc3, other=0)

    o_c0 = (i_tc0 + o_i).to(tl.int64)
    o_c1 = (i_tc1 + o_i).to(tl.int64)
    o_c2 = (i_tc2 + o_i).to(tl.int64)
    o_c3 = (i_tc3 + o_i).to(tl.int64)

    ############################################################################
    # Step 1: 10 lower-triangular [BC, BC] blocks of K @ K^T
    ############################################################################

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(
            k, (T, K), (Hg * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(
                k, (T, K), (Hg * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (Hg * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    ############################################################################
    # Step 2: gate decay  exp(G_row - G_col)
    ############################################################################

    b_A00 *= safe_exp(b_g0[:, None] - b_g0[None, :])
    b_A11 *= safe_exp(b_g1[:, None] - b_g1[None, :])
    b_A22 *= safe_exp(b_g2[:, None] - b_g2[None, :])
    b_A33 *= safe_exp(b_g3[:, None] - b_g3[None, :])
    b_A10 *= safe_exp(b_g1[:, None] - b_g0[None, :])
    b_A20 *= safe_exp(b_g2[:, None] - b_g0[None, :])
    b_A21 *= safe_exp(b_g2[:, None] - b_g1[None, :])
    b_A30 *= safe_exp(b_g3[:, None] - b_g0[None, :])
    b_A31 *= safe_exp(b_g3[:, None] - b_g1[None, :])
    b_A32 *= safe_exp(b_g3[:, None] - b_g2[None, :])

    ############################################################################
    # Step 3: proper-ancestor mask (replaces the chain strict tril) + beta
    ############################################################################

    m_a00 = ((b_anc0[:, None] >> o_c0[None, :]) & 1) != 0
    m_a11 = ((b_anc1[:, None] >> o_c1[None, :]) & 1) != 0
    m_a22 = ((b_anc2[:, None] >> o_c2[None, :]) & 1) != 0
    m_a33 = ((b_anc3[:, None] >> o_c3[None, :]) & 1) != 0
    m_a10 = ((b_anc1[:, None] >> o_c0[None, :]) & 1) != 0
    m_a20 = ((b_anc2[:, None] >> o_c0[None, :]) & 1) != 0
    m_a21 = ((b_anc2[:, None] >> o_c1[None, :]) & 1) != 0
    m_a30 = ((b_anc3[:, None] >> o_c0[None, :]) & 1) != 0
    m_a31 = ((b_anc3[:, None] >> o_c1[None, :]) & 1) != 0
    m_a32 = ((b_anc3[:, None] >> o_c2[None, :]) & 1) != 0

    b_A00 = tl.where(m_a00, b_A00, 0.0) * b_b0[:, None]
    b_A11 = tl.where(m_a11, b_A11, 0.0) * b_b1[:, None]
    b_A22 = tl.where(m_a22, b_A22, 0.0) * b_b2[:, None]
    b_A33 = tl.where(m_a33, b_A33, 0.0) * b_b3[:, None]
    b_A10 = tl.where(m_a10, b_A10, 0.0) * b_b1[:, None]
    b_A20 = tl.where(m_a20, b_A20, 0.0) * b_b2[:, None]
    b_A21 = tl.where(m_a21, b_A21, 0.0) * b_b2[:, None]
    b_A30 = tl.where(m_a30, b_A30, 0.0) * b_b3[:, None]
    b_A31 = tl.where(m_a31, b_A31, 0.0) * b_b3[:, None]
    b_A32 = tl.where(m_a32, b_A32, 0.0) * b_b3[:, None]

    ############################################################################
    # Step 4: forward substitution on diagonal blocks -> (I + A_diag)^{-1}
    ############################################################################

    m_I = o_i[:, None] == o_i[None, :]

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    ############################################################################
    # Step 5: block merge -> full (I + A)^{-1}
    ############################################################################

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=_MERGE_DOT_PRECISION),
        b_Ai00,
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=_MERGE_DOT_PRECISION),
        b_Ai11,
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=_MERGE_DOT_PRECISION),
        b_Ai22,
        input_precision=_MERGE_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A21, b_Ai10, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai21, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A31, b_Ai10, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai20, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )

    ############################################################################
    # Step 6: store full (I + A)^{-1}
    ############################################################################

    p_A00 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_A10 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A22 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
    )
    p_A30 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
    )
    p_A33 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
    )

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def tree_verify_o_kernel(
    q,
    k,
    w,
    u,
    g,
    ancestor_masks,
    h0_source,
    h0_indices,
    o,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Fused verify output:

      o_i = scale * ( exp(G_i) * q_i @ h0
                      + sum_{j in path(i)} (q_i . k_j) exp(G_i - G_j) (u_j - w_j @ h0) )

    Reads h0 straight from the state pool; writes nothing but o.
    """
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    w += (bos * H + i_h) * K
    u += (bos * H + i_h) * V
    o += (bos * H + i_h) * V

    h0_idx = tl.load(h0_indices + i_b).to(tl.int64)
    # Negative index = padded request (PAD_SLOT_ID); read slot 0 and zero the
    # contribution so the pointer arithmetic stays in bounds.
    h0_valid = (h0_idx >= 0).to(tl.float32)
    h0 = h0_source + (tl.maximum(h0_idx, 0) * H + i_h) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_v_new = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, 0), (BK, BT), (0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_h0 = tl.make_block_ptr(
            h0, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_h0 = (tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32) * h0_valid).to(
            b_q.dtype
        )

        b_o += tl.dot(b_q, tl.trans(b_h0))
        b_A += tl.dot(b_q, b_k)
        b_v_new += tl.dot(b_w, tl.trans(b_h0))

    o_t = tl.arange(0, BT)
    m_t = o_t < T
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (0,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_anc = tl.load(ancestor_masks + i_b * BT + o_t, mask=m_t, other=0)

    b_o = b_o * exp(b_g)[:, None]
    b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])
    m_path = ((b_anc[:, None] >> o_t[None, :].to(tl.int64)) & 1) != 0
    m_path |= o_t[:, None] == o_t[None, :]
    b_A = tl.where(m_path, b_A, 0.0)

    p_u = tl.make_block_ptr(u, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    b_u = tl.load(p_u, boundary_check=(0, 1))
    b_v_new = (b_u - b_v_new).to(b_u.dtype)

    b_o = (b_o + tl.dot(b_A.to(b_v_new.dtype), b_v_new)) * scale

    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))



@triton.jit(do_not_specialize=["T"])
def tree_verify_o_neumann_kernel(
    q,
    k,
    v,
    beta,
    g,
    ancestor_masks,
    h0_source,
    h0_indices,
    o,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NEUMANN_TERMS: tl.constexpr,
):
    """Bole's parallel tree recurrence: no explicit inverse, no A_inv round trip.

    arXiv 2608.01651 Theorem 1 writes the whole tree verify in closed form

        O = DP Q Spre + C (I+G)^-1 Dbeta (V - DP K Spre)
        G = Dbeta DP [(K K^T) . M-] DP^-1        (M- = STRICT ancestors)
        C = DP [(Q K^T) . M+] DP^-1              (M+ = ancestors + self)

    G is nonzero only on ancestor pairs, so it is nilpotent in the tree depth:
    G^(d+1) = 0. Their Lemma 3 therefore replaces the inverse with a FINITE
    Neumann series that is exact, not approximate:

        (I+G)^-1 = sum_{m=0..d} (-G)^m

    evaluated as the recurrence Z(0) = R, Z(m+1) = -G Z(m), U = sum_m Z(m).

    Why this is faster than what we had:
      * `tree_kkt_solve_kernel` inverts by FORWARD SUBSTITUTION -- a serial
        row-at-a-time loop that never reaches a tensor core -- and then writes a
        [T, T] A_inv per (batch, head) to HBM for the output kernel to read back.
      * Each -G Z(m) here is a dense MMA, so the solve becomes d+1 tensor-core
        ops and G, Z and the output tile never leave the CTA. The paper reports
        occupancy up 3.6-7.9x, state materialisation being 38%/86% of serial
        verify time on A100/GB10, and 3.4-7.5x per-layer GDN latency (Table VII).

    NEUMANN_TERMS is the tree's maximum depth: a property of the drafter's block
    size, not of the data, so it stays a compile-time constant and the loop
    unrolls. MORE terms than the depth is harmless -- the extra powers of G are
    exactly zero -- FEWER is wrong, so the caller passes the bound.
    """
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V

    h0_idx = tl.load(h0_indices + i_b).to(tl.int64)
    h0_valid = (h0_idx >= 0).to(tl.float32)
    h0 = h0_source + (tl.maximum(h0_idx, 0) * H + i_h) * V * K

    o_t = tl.arange(0, BT)
    m_t = o_t < T

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_qk = tl.zeros([BT, BT], dtype=tl.float32)
    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    b_kh0 = tl.zeros([BT, BV], dtype=tl.float32)

    # One pass over K builds every second-order term the closed form needs:
    # Q h0^T (the Spre readout), Q K^T (C), K K^T (G) and K h0^T (the residual).
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_kt = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_h0 = tl.make_block_ptr(
            h0, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kt = tl.load(p_kt, boundary_check=(0, 1))
        b_h0 = (tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32) * h0_valid).to(
            b_q.dtype
        )
        b_o += tl.dot(b_q, tl.trans(b_h0))
        b_qk += tl.dot(b_q, tl.trans(b_kt))
        b_kk += tl.dot(b_kt, tl.trans(b_kt))
        b_kh0 += tl.dot(b_kt, tl.trans(b_h0))

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (0,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (0,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_anc = tl.load(ancestor_masks + i_b * BT + o_t, mask=m_t, other=0)

    # M- : j is a STRICT ancestor of i. M+ additionally admits the diagonal.
    m_strict = ((b_anc[:, None] >> o_t[None, :].to(tl.int64)) & 1) != 0

    # G = Dbeta DP [(K K^T) . M-] DP^-1 -- exactly the matrix that
    # tree_kkt_solve_kernel builds as A before inverting it.
    #
    # REGISTER PRESSURE: build G in place over b_kk and narrow it to the matmul
    # dtype immediately. Keeping b_kk, a separate fp32 b_G, b_qk, b_o, b_kh0, b_z
    # and b_u all live is seven [BT,BT]/[BT,BV] fp32 tiles -- about 110KB per CTA,
    # which spills at any sane warp count and made a first version slower than the
    # two-kernel reference it replaces.
    # Load v FIRST and keep its native dtype: the Neumann matmuls must run at the
    # reference's bf16 precision. An earlier version cast G and Z to b_v.dtype
    # AFTER b_v had already been promoted to fp32, so every tl.dot in the series
    # ran in fp32 -- off the tensor cores, in the innermost loop of the kernel.
    # That single line, not the algorithm, is what made the series look slow.
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    b_v_raw = tl.load(p_v, boundary_check=(0, 1))
    b_v = b_v_raw.to(tl.float32)

    b_kk = b_kk * safe_exp(b_g[:, None] - b_g[None, :])
    b_Gc = (tl.where(m_strict, b_kk, 0.0) * b_beta[:, None]).to(b_v_raw.dtype)

    # R = Dbeta (V - DP K Spre)
    b_rhs = b_beta[:, None] * (b_v - exp(b_g)[:, None] * b_kh0)

    # Finite Neumann, entirely on chip, accumulating in place. Kept at the
    # reference's matmul precision: the unfused path materialises u in bf16 and
    # does the final product in bf16, so an fp32 tl.dot here would be both slower
    # and numerically different from the kernel this is gated against.
    b_z = b_rhs.to(b_v_raw.dtype)
    for _ in tl.static_range(NEUMANN_TERMS):
        b_z = (-tl.dot(b_Gc, b_z)).to(b_v_raw.dtype)
        b_rhs += b_z
    b_v_new = b_rhs.to(b_v_raw.dtype)

    m_path = m_strict | (o_t[:, None] == o_t[None, :])

    b_o = b_o * exp(b_g)[:, None]
    b_qk = b_qk * safe_exp(b_g[:, None] - b_g[None, :])
    b_qk = tl.where(m_path, b_qk, 0.0)

    b_o = (b_o + tl.dot(b_qk.to(b_v_new.dtype), b_v_new)) * scale

    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def tree_verify_o_replay_kernel(
    q,
    k,
    v,
    beta,
    g,
    ancestor_masks,
    h0_source,
    h0_indices,
    rp_d,          # [num_slots, HV, L, V] ring buffer: correction terms u
    rp_k,          # [num_slots, H,  L, K] ring buffer: keys
    rp_g,          # [num_slots, HV, L]    ring buffer: log decays (fp32)
    rp_pos,        # [B] int32 ring cursor: number of valid entries
    o,
    scale,
    T,
    L_true,        # actual ring length; block extent below is padded separately
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NEUMANN_TERMS: tl.constexpr,
):
    """Tree verify that REPLAYS a ring buffer instead of reading a folded state.

    ReplaySSM (Dao AI Lab, 2026) caches SSM *inputs* rather than state. The commit
    after each verify step appends the accepted path's (u, k, g) to a ring buffer
    and only materialises a full checkpoint every L steps, so the per-step 8dn
    state store becomes a ~2(d+n+1) append. The post is explicit that this is what
    the tree case needs: "the ring buffer with indexing is also important for
    tree-based speculative decoding, where accepted tokens are no longer
    contiguous in the buffer", and that during verification "each draft query reads
    from the same checkpoint state and the same buffer window -- the only
    difference across draft positions is the causal mask."

    The pre-tree state is therefore not a tensor we load; it is

        Spre = P_h * S0 + sum_{j<h} (P_h / P_j) * k_j u_j^T

    with P_j the cumulative decay through buffer entry j. Every place the folded
    kernel used Spre, this one uses the checkpoint term plus a replay term:

        Q Spre^T = P_h * (Q S0^T) + [(Q K_buf^T) * w] U_buf
        K Spre^T = P_h * (K S0^T) + [(K K_buf^T) * w] U_buf,   w_j = P_h / P_j

    Both replay terms are [BT, L] @ [L, BV] with L <= 16, so they are small next to
    the [BT, K] @ [K, BV] checkpoint product they supplement.
    """
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V

    h0_idx = tl.load(h0_indices + i_b).to(tl.int64)
    h0_valid = (h0_idx >= 0).to(tl.float32)
    slot = tl.maximum(h0_idx, 0)
    h0 = h0_source + (slot * H + i_h) * V * K

    o_t = tl.arange(0, BT)
    o_l = tl.arange(0, L)
    m_t = o_t < T

    # Ring contents. `rp_pos` is the number of committed entries, so entries
    # [0, rp_pos) are live and the rest are stale -- masking on the cursor is what
    # makes a rollback "purely a pointer move" with no buffer rewrite.
    n_valid = tl.load(rp_pos + i_b).to(tl.int32)
    # The block extent L is padded to a tensor-core friendly width; validity is
    # bounded by BOTH the live cursor and the real ring length.
    m_l = (o_l < n_valid) & (o_l < L_true)

    p_rg = rp_g + (slot * H + i_h) * L_true + o_l
    b_rg = tl.load(p_rg, mask=m_l & (o_l < L_true), other=0.0).to(tl.float32)
    # Suffix decay: P_h / P_j = exp(sum_{i>j} g_i). Computed as a masked scan so
    # the whole window is one pass and no cumulative state crosses iterations.
    g_tot = tl.sum(tl.where(m_l, b_rg, 0.0))
    g_pre = tl.cumsum(tl.where(m_l, b_rg, 0.0), axis=0)
    # NOT safe_exp here. safe_exp(x) = exp(where(x <= 0, x, -inf)) returns ZERO for
    # any positive argument -- it exists to mask non-ancestor pairs. These decays
    # are mathematically <= 0, but for the most recent entry g_tot - g_pre is
    # exactly 0 in exact arithmetic and lands epsilon-POSITIVE after a float
    # cumsum, so safe_exp silently dropped the newest buffer entry. With a 4-entry
    # window that is a ~25% error in the replay term, which is what the identity
    # test caught. Clamp instead of masking.
    w_j = tl.where(m_l, exp(tl.minimum(g_tot - g_pre, 0.0)), 0.0)
    p_h = exp(tl.minimum(g_tot, 0.0))                       # decay on the checkpoint

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_qk = tl.zeros([BT, BT], dtype=tl.float32)
    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    b_kh0 = tl.zeros([BT, BV], dtype=tl.float32)
    b_qkb = tl.zeros([BT, L], dtype=tl.float32)             # Q @ K_buf^T
    b_kkb = tl.zeros([BT, L], dtype=tl.float32)             # K @ K_buf^T

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_kt = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_h0 = tl.make_block_ptr(
            h0, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kt = tl.load(p_kt, boundary_check=(0, 1))
        b_h0 = (tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32) * h0_valid).to(
            b_q.dtype
        )
        b_o += tl.dot(b_q, tl.trans(b_h0))
        b_qk += tl.dot(b_q, tl.trans(b_kt))
        b_kk += tl.dot(b_kt, tl.trans(b_kt))
        b_kh0 += tl.dot(b_kt, tl.trans(b_h0))

        # Replay keys for this K slice. rp_k is [slot, H, L, K]; the head group
        # index matches q/k, not the value head.
        p_rk = tl.make_block_ptr(
            rp_k + (slot * Hg + i_h // (H // Hg)) * L_true * K,
            (L_true, K), (K, 1), (0, i_k * BK), (L, BK), (1, 0),
        )
        b_rk = tl.load(p_rk, boundary_check=(0, 1))
        b_qkb += tl.dot(b_q, tl.trans(b_rk))
        b_kkb += tl.dot(b_kt, tl.trans(b_rk))

    # Replay values for this value tile: [L, BV].
    p_rd = tl.make_block_ptr(
        rp_d + (slot * H + i_h) * L_true * V,
        (L_true, V), (V, 1), (0, i_v * BV), (L, BV), (1, 0),
    )
    b_rd = tl.load(p_rd, boundary_check=(0, 1))

    # Spre products = decayed checkpoint + decay-weighted replay.
    b_qkb = tl.where(m_l[None, :], b_qkb * w_j[None, :], 0.0)
    b_kkb = tl.where(m_l[None, :], b_kkb * w_j[None, :], 0.0)
    b_o = b_o * p_h + tl.dot(b_qkb.to(b_rd.dtype), b_rd)
    b_kh0 = b_kh0 * p_h + tl.dot(b_kkb.to(b_rd.dtype), b_rd)

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (0,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (0,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_anc = tl.load(ancestor_masks + i_b * BT + o_t, mask=m_t, other=0)

    m_strict = ((b_anc[:, None] >> o_t[None, :].to(tl.int64)) & 1) != 0

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    b_v_raw = tl.load(p_v, boundary_check=(0, 1))
    b_v = b_v_raw.to(tl.float32)

    b_kk = b_kk * safe_exp(b_g[:, None] - b_g[None, :])
    b_Gc = (tl.where(m_strict, b_kk, 0.0) * b_beta[:, None]).to(b_v_raw.dtype)

    b_rhs = b_beta[:, None] * (b_v - exp(b_g)[:, None] * b_kh0)

    b_z = b_rhs.to(b_v_raw.dtype)
    for _ in tl.static_range(NEUMANN_TERMS):
        b_z = (-tl.dot(b_Gc, b_z)).to(b_v_raw.dtype)
        b_rhs += b_z
    b_v_new = b_rhs.to(b_v_raw.dtype)

    m_path = m_strict | (o_t[:, None] == o_t[None, :])
    b_o = b_o * exp(b_g)[:, None]
    b_qk = b_qk * safe_exp(b_g[:, None] - b_g[None, :])
    b_qk = tl.where(m_path, b_qk, 0.0)
    b_o = (b_o + tl.dot(b_qk.to(b_v_new.dtype), b_v_new)) * scale

    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def tree_verify_o_fused_kernel(
    q,
    k,
    v,
    beta,
    A_inv,
    g,
    ancestor_masks,
    h0_source,
    h0_indices,
    o,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """tree_verify_o_kernel with recompute_w_u folded in.

    The unfused path materialises two [B, T, H, *] tensors that exist only to be
    read by the output stage:

        u = A_inv @ (v * beta)                    [B, T, H, V]
        w = A_inv @ (k * beta * exp(g_path))      [B, T, H, K]

    and the output needs only their combination, which factors:

        u - w @ h0 = A_inv @ [ beta * (v - exp(g_path) * (k @ h0)) ]

    so the bracket is formed per value tile on chip and neither w nor u is ever
    written. At B=32, T=18, H=48, K=V=128 that removes ~14 MB of live
    intermediates per GDN layer, x48 layers -- which is the point: CUDA-graph
    capture memory scales with live per-token intermediates (measured ~4.5 MB per
    batch x draft token), and that memory is what makes c64 unservable and limits
    how many graph capacities we can afford.

    Same grid and same output as tree_verify_o_kernel; A_inv is the [T, BT]
    factor tree_kkt_solve_kernel already produces.
    """
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    A_inv += (bos * H + i_h) * BT
    o += (bos * H + i_h) * V

    h0_idx = tl.load(h0_indices + i_b).to(tl.int64)
    h0_valid = (h0_idx >= 0).to(tl.float32)
    h0 = h0_source + (tl.maximum(h0_idx, 0) * H + i_h) * V * K

    o_t = tl.arange(0, BT)
    m_t = o_t < T

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_qk = tl.zeros([BT, BT], dtype=tl.float32)
    b_kh0 = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_kt = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (0, i_k * BK), (BT, BK), (1, 0))
        p_h0 = tl.make_block_ptr(
            h0, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kt = tl.load(p_kt, boundary_check=(0, 1))
        b_h0 = (tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32) * h0_valid).to(
            b_q.dtype
        )
        b_o += tl.dot(b_q, tl.trans(b_h0))
        b_qk += tl.dot(b_q, tl.trans(b_kt))
        # k @ h0^T for this value tile -- replaces the unfused w @ h0.
        b_kh0 += tl.dot(b_kt, tl.trans(b_h0))

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (0,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (0,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_anc = tl.load(ancestor_masks + i_b * BT + o_t, mask=m_t, other=0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

    # rhs = beta * (v - exp(g_path) * (k @ h0))
    b_rhs = b_beta[:, None] * (b_v - exp(b_g)[:, None] * b_kh0)

    # (u - w @ h0) = A_inv @ rhs, entirely on chip
    p_Ai = tl.make_block_ptr(A_inv, (T, BT), (H * BT, 1), (0, 0), (BT, BT), (1, 0))
    b_Ai = tl.load(p_Ai, boundary_check=(0, 1))
    # Cast back to the input dtype: the unfused path materialises u in bf16 and
    # does the final (masked QK) @ (u - w h0) product in bf16, so keeping this in
    # fp32 would both change numerics relative to the reference and force a slow
    # fp32 tl.dot.
    b_v_new = tl.dot(b_Ai, b_rhs.to(b_Ai.dtype)).to(b_Ai.dtype)

    b_o = b_o * exp(b_g)[:, None]
    b_qk = b_qk * safe_exp(b_g[:, None] - b_g[None, :])
    m_path = ((b_anc[:, None] >> o_t[None, :].to(tl.int64)) & 1) != 0
    m_path |= o_t[:, None] == o_t[None, :]
    b_qk = tl.where(m_path, b_qk, 0.0)

    b_o = (b_o + tl.dot(b_qk.to(b_v_new.dtype), b_v_new)) * scale

    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_tree_verify_replay(
    q,
    k,
    v,
    g,
    beta,
    ancestor_masks,
    initial_state_source,
    initial_state_indices,
    replay_d,
    replay_k,
    replay_g,
    replay_pos,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    max_tree_depth=None,
    num_warps=4,
    num_stages=1,
    BV=None,
):
    """Tree verify against a ReplaySSM checkpoint + ring buffer.

    Same output as `chunk_gated_delta_rule_tree_verify_neumann`, but the pre-tree
    state is reconstructed from (checkpoint, buffer) rather than read as a folded
    tensor. That is what lets the COMMIT append 2(d+n+1) bytes per step instead of
    storing 8dn -- the actual saving ReplaySSM claims.

    replay_d/k/g are the ring buffers already allocated by the decode path
    (`[num_slots, HV, L, V]`, `[num_slots, H, L, K]`, `[num_slots, HV, L]`);
    replay_pos is the per-request count of live entries.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = TREE_CHUNK_SIZE
    assert T <= BT, f"chunk tree verify supports at most {BT} draft tokens, got {T}"
    if ancestor_masks.dim() == 3:
        assert ancestor_masks.shape[-1] == 1 and ancestor_masks.shape[-2] == BT
        ancestor_masks = ancestor_masks[..., 0]
    assert q.dtype != torch.float32, "use bfloat16/float16 inputs"
    L = replay_g.shape[-1]

    if scale is None:
        scale = K**-0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()

    g_path = torch.empty_like(g, dtype=torch.float32)
    tree_path_cumsum_kernel[(B,)](
        g=g, g_path=g_path, ancestor_masks=ancestor_masks, T=T, H=H, BT=BT,
        BH=max(16, triton.next_power_of_2(H)),
    )

    terms = int(max_tree_depth) if max_tree_depth is not None else NEUMANN_DEFAULT_TERMS
    terms = max(1, min(terms, int(T) - 1))
    if BV is None:
        BV = V if V <= 128 else 64

    o = torch.empty_like(v)
    grid = (triton.cdiv(V, BV), B * H)
    tree_verify_o_replay_kernel[grid](
        q=q, k=k, v=v, beta=beta, g=g_path, ancestor_masks=ancestor_masks,
        h0_source=initial_state_source, h0_indices=initial_state_indices,
        rp_d=replay_d, rp_k=replay_k, rp_g=replay_g, rp_pos=replay_pos,
        o=o, scale=scale, T=T, L_true=L, H=H, Hg=Hg, K=K, V=V,
        # Pad the buffer axis to 64: a tl.dot with N=16 is below the tensor-core
        # tile and was measurably wrong on this target (20% error on the replay
        # term while the checkpoint term was exact). Masking keeps it correct.
        L=max(64, triton.next_power_of_2(L)), BT=BT, BK=64, BV=BV,
        NEUMANN_TERMS=terms, num_warps=num_warps, num_stages=num_stages,
    )
    return o


def chunk_gated_delta_rule_tree_verify_neumann(
    q,
    k,
    v,
    g,
    beta,
    ancestor_masks,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    max_tree_depth=None,
    num_warps=4,
    num_stages=1,
    BV=None,
):
    """Bole's two-stage tree verify (arXiv 2608.01651).

    Drops BOTH middle stages of the existing pipeline:

        cumsum -> [kkt solve -> recompute_w_u] -> output      (before)
        cumsum -> output                                      (here)

    The KKT solve inverted (I+G) by serial forward substitution and staged a
    [B, T, H, BT] A_inv through HBM purely so the output kernel could read it
    back. G is nilpotent in the tree depth, so the inverse is a finite Neumann
    series the output CTA can evaluate itself with tensor-core MMAs.

    `max_tree_depth` bounds the series. It is a config property (the drafter's
    block size), not data, so it stays constexpr. Defaults to T-1, which is
    always correct -- a chain -- but costs terms that are identically zero.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    # BT is the fixed chunk extent, NOT next_power_of_2(T). The kernels index
    # ancestor_masks with a stride of BT per batch, so deriving BT from T silently
    # reads the wrong rows for any T that is not already TREE_CHUNK_SIZE -- which
    # is why an earlier version passed at T=64 and failed at T=18 and T=32.
    BT = TREE_CHUNK_SIZE
    assert T <= BT, f"chunk tree verify supports at most {BT} draft tokens, got {T}"
    if ancestor_masks.dim() == 3:
        assert ancestor_masks.shape[-1] == 1 and ancestor_masks.shape[-2] == BT
        ancestor_masks = ancestor_masks[..., 0]
    assert q.dtype != torch.float32, "use bfloat16/float16 inputs"

    if scale is None:
        scale = K**-0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    g_path = torch.empty_like(g, dtype=torch.float32)
    tree_path_cumsum_kernel[(B,)](
        g=g,
        g_path=g_path,
        ancestor_masks=ancestor_masks,
        T=T,
        H=H,
        BT=BT,
        BH=max(16, triton.next_power_of_2(H)),
    )

    # Term count. Bole specifies d+1 for exactness, but the series converges far
    # sooner here and the extra terms are pure cost: measured on production-shaped
    # inputs (q/k l2-normed, so max|G| ~ 0.13) the error hits the bf16 floor of
    # 2.6e-3 at TWO terms and does not improve through 15. NEUMANN_DEFAULT_TERMS
    # keeps a 2x margin over that saturation point and is still the fastest verify
    # path measured; raise it toward the depth if a drafter ever produces trees
    # with |G| near its 1.0 bound.
    terms = int(max_tree_depth) if max_tree_depth is not None else NEUMANN_DEFAULT_TERMS
    terms = max(1, min(terms, int(T) - 1))
    # G is value-tile independent, so a single tile builds it once per (batch,head)
    # instead of once per tile -- worth 1.55x over base against 1.13x at BV=64.
    if BV is None:
        BV = V if V <= 128 else 64

    o = torch.empty_like(v)
    # G does NOT depend on the value tile, but this kernel builds it itself rather
    # than loading a precomputed A_inv -- so every extra value tile rebuilds it.
    # With V=128, BV=64 costs two constructions per (batch, head) where the
    # reference's KKT kernel pays one. BV=V removes that redundancy at the cost of
    # wider live tiles.
    grid = (triton.cdiv(V, BV), B * H)
    tree_verify_o_neumann_kernel[grid](
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g_path,
        ancestor_masks=ancestor_masks,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        o=o,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=BV,
        NEUMANN_TERMS=terms,
        # Exposed for profiling: this kernel carries more live tile state than the
        # reference output kernel (it builds G itself instead of loading A_inv), so
        # the reference's num_warps=4 is not automatically the right choice here.
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def chunk_gated_delta_rule_tree_verify_fused(
    q,
    k,
    v,
    g,
    beta,
    ancestor_masks,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Three-stage plain-path verify: cumsum -> KKT solve -> fused output.

    Identical semantics to chunk_gated_delta_rule_tree_verify, minus the
    recompute_w_u_fwd stage: the output kernel forms (u - w @ h0) on chip from the
    KKT factor, so the [B,T,H,K] w and [B,T,H,V] u tensors are never allocated.

    The point is CUDA-graph capture memory, which scales with live per-token
    intermediates (~4.5 MB per batch x draft token measured from server logs).
    Whether that translates into a smaller capture line is an END-TO-END question
    for the server log; do not infer it from a microbenchmark.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[2], v.shape[3]
    BT = tree_mask_capacity(T)

    if ancestor_masks.dim() == 3:
        assert ancestor_masks.shape[-1] == 1 and ancestor_masks.shape[-2] == BT
        ancestor_masks = ancestor_masks[..., 0]
    assert q.dtype != torch.float32, "use bfloat16/float16 inputs"

    if scale is None:
        scale = K**-0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    g_path = torch.empty_like(g, dtype=torch.float32)
    tree_path_cumsum_kernel[(B,)](
        g=g,
        g_path=g_path,
        ancestor_masks=ancestor_masks,
        T=T,
        H=H,
        BT=BT,
        BH=max(16, triton.next_power_of_2(H)),
    )

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    tree_kkt_solve_kernel[(B * H,)](
        k=k,
        g=g_path,
        beta=beta,
        ancestor_masks=ancestor_masks,
        A=A,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BC=16,
    )

    o = torch.empty_like(v)
    grid = (triton.cdiv(V, 64), B * H)
    tree_verify_o_fused_kernel[grid](
        q=q,
        k=k,
        v=v,
        beta=beta,
        A_inv=A,
        g=g_path,
        ancestor_masks=ancestor_masks,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        o=o,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
        num_warps=4,
        # Matches the unfused kernel: the SM100 pipeliner crashes on this
        # multi-dot K-loop and the loop is only K/BK=2 iterations.
        num_stages=1,
    )
    return o


@triton.jit(do_not_specialize=["T"])
def tree_verify_state_advance_kernel(
    k_stash,
    v_stash,
    g_stash,
    beta_stash,
    ssm_states,
    cache_indices,
    ancestor_masks,
    last_correct_steps,
    track_steps,
    track_slots,
    T,
    stride_kl,
    stride_vl,
    stride_gl,
    stride_bl,
    stride_sl,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    NW: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    """Commit SSM states after verification by replaying the delta rule along
    each request's accept path (root -> last correct draft).

    The accept path is recovered GPU-side from the leaf's proper-ancestor
    bitset words: path = ancestor_masks[leaf] | (1 << leaf), word w covering
    tree slots [64w, 64w + 64). Ancestors of a node are totally ordered and
    topologically sorted, so iterating tree slots in increasing index order
    replays the path root-first.

    Replaces the recurrent verify kernel's per-token intermediate state cache
    (T full states per layer per request) with one state read + one write.

    k_stash must already be L2-normalized; g/beta are the gating outputs.
    Stash layout per layer l (leading stride stride_*l):
      k_stash: [B, T, Hg*K], v_stash: [B, T, H*V], g/beta_stash: [B, T, H]
    ssm_states: [L, pool, H, V, K] with per-layer stride stride_sl.
    """
    i_l, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // H, i_nh % H
    i_h = i_hv // (H // Hg)

    slot = tl.load(cache_indices + i_n).to(tl.int64)
    if slot < 0:
        # Padded request (PAD_SLOT_ID): nothing to commit.
        return

    leaf = tl.load(last_correct_steps + i_n).to(tl.int64)
    leaf = tl.minimum(tl.maximum(leaf, 0), T - 1)
    o_w = tl.arange(0, NW)
    b_anc = tl.load(ancestor_masks + (i_n * BT + leaf) * NW + o_w)
    path_words = b_anc | tl.where(
        o_w == (leaf // 64), tl.full((), 1, tl.int64) << (leaf % 64), 0
    )

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_h = (
        ssm_states
        + i_l * stride_sl
        + slot * H * K * V
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    if HAS_TRACK:
        track_step = tl.load(track_steps + i_n).to(tl.int64)
        track_slot = tl.load(track_slots + i_n).to(tl.int64)
    else:
        track_step = -1
        track_slot = 0

    p_k = k_stash + i_l * stride_kl + i_n * T * Hg * K + i_h * K + o_k
    p_v = v_stash + i_l * stride_vl + i_n * T * H * V + i_hv * V + o_v
    p_g = g_stash + i_l * stride_gl + i_n * T * H + i_hv
    p_b = beta_stash + i_l * stride_bl + i_n * T * H + i_hv

    for t in range(0, T):
        word = tl.sum(tl.where(o_w == (t // 64), path_words, 0))
        on_path = ((word >> (t % 64)) & 1) != 0
        if on_path:
            b_k = tl.load(p_k + t * Hg * K, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v + t * H * V, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g + t * H).to(tl.float32)
            b_beta = tl.load(p_b + t * H).to(tl.float32)

            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            if HAS_TRACK:
                if t == track_step:
                    p_track = (
                        ssm_states
                        + i_l * stride_sl
                        + track_slot * H * K * V
                        + i_hv * K * V
                        + o_v[None, :] * K
                        + o_k[:, None]
                    )
                    tl.store(
                        p_track, b_h.to(p_track.dtype.element_ty), mask=mask_h
                    )

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


@triton.jit
def tree_verify_state_append_kernel(
    k_stash,
    v_stash,
    g_stash,
    beta_stash,
    ssm_states,          # checkpoint S0, written ONLY on flush
    rp_d,                # [num_slots, HV, L, V] ring: corrections u
    rp_k,                # [num_slots, H,  L, K] ring: keys
    rp_g,                # [num_slots, HV, L]    ring: log decays (fp32)
    rp_pos,              # [B] int32 live-entry count, updated device-side
    cache_indices,
    ancestor_masks,
    last_correct_steps,
    T,
    L_ring,
    stride_kl,
    stride_vl,
    stride_gl,
    stride_bl,
    stride_sl,
    stride_rdl,          # per-LAYER stride of the ring buffers. The grid iterates
    stride_rkl,          # layers, so omitting these writes every layer's inputs
    stride_rgl,          # into layer 0's ring.
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    NW: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_WINDOW: tl.constexpr,
):
    """Commit the accepted path by APPENDING inputs, not storing state.

    ReplaySSM's saving is the store: the baseline writes the full [K, V] state
    every step (8dn bytes), this appends (u, k, g) per accepted node
    (~2(d+n+1) bytes) and only materialises a checkpoint on flush.

    Flush policy follows the paper's speculative rule rather than the standard one.
    Standard decoding flushes when `h + 1 > L`. Under speculation the post flushes
    ONE WINDOW EARLY, `h + 2T > L`, which "guarantees >= T free slots per step and
    prevents window truncation on acceptance" -- without it a step that accepts a
    full window would need to split its append across a flush boundary.

    GDN cannot cache raw v (the post's gotcha 3): u_t = beta_t (v_t - S_{t-1} k_t)
    needs the running state, so the state IS materialised in registers here. That
    is inherent to the delta rule and is why only the store, not the load, is
    saved.

    rp_pos is updated on device. Sending accepted counts to the host would add a
    sync every step, which the post calls out explicitly as something to avoid.
    """
    i_l, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // H, i_nh % H
    i_h = i_hv // (H // Hg)

    slot = tl.load(cache_indices + i_n).to(tl.int64)
    leaf = tl.load(last_correct_steps + i_n).to(tl.int64)

    rp_d += i_l * stride_rdl
    rp_k += i_l * stride_rkl
    rp_g += i_l * stride_rgl

    o_w = tl.arange(0, NW)
    b_anc = tl.load(ancestor_masks + (i_n * BT + leaf) * NW + o_w)
    path_words = b_anc | tl.where(
        o_w == (leaf // 64), tl.full((), 1, tl.int64) << (leaf % 64), 0
    )

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_h = (
        ssm_states + i_l * stride_sl + slot * H * K * V + i_hv * K * V
        + o_v[None, :] * K + o_k[:, None]
    )
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    # Replay the live window onto the checkpoint to recover the current state.
    h_live = tl.load(rp_pos + i_n).to(tl.int32)
    for j in range(0, h_live):
        rg = tl.load(rp_g + (slot * H + i_hv) * L_ring + j).to(tl.float32)
        rk = tl.load(rp_k + ((slot * Hg + i_h) * L_ring + j) * K + o_k,
                     mask=mask_k, other=0).to(tl.float32)
        rd = tl.load(rp_d + ((slot * H + i_hv) * L_ring + j) * V + o_v,
                     mask=mask_v, other=0).to(tl.float32)
        b_h = b_h * tl.exp(rg) + rk[:, None] * rd[None, :]

    p_k = k_stash + i_l * stride_kl + i_n * T * Hg * K + i_h * K + o_k
    p_v = v_stash + i_l * stride_vl + i_n * T * H * V + i_hv * V + o_v
    p_g = g_stash + i_l * stride_gl + i_n * T * H + i_hv
    p_b = beta_stash + i_l * stride_bl + i_n * T * H + i_hv

    # Count what this step will append, so the flush decision is made before any
    # write and the window is never split.
    n_acc = 0
    for t in range(0, T):
        word = tl.sum(tl.where(o_w == (t // 64), path_words, 0))
        if ((word >> (t % 64)) & 1) != 0:
            n_acc += 1

    will_flush = (h_live + n_acc + SPEC_WINDOW) > L_ring

    w = h_live
    for t in range(0, T):
        word = tl.sum(tl.where(o_w == (t // 64), path_words, 0))
        on_path = ((word >> (t % 64)) & 1) != 0
        if on_path:
            b_k = tl.load(p_k + t * Hg * K, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v + t * H * V, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g + t * H).to(tl.float32)
            b_beta = tl.load(p_b + t * H).to(tl.float32)

            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            if not will_flush:
                # Append. Only the value-tile owner writes k and g, which are not
                # tiled over V, so they are not written once per tile.
                tl.store(rp_d + ((slot * H + i_hv) * L_ring + w) * V + o_v,
                         b_v.to(rp_d.dtype.element_ty), mask=mask_v)
                if i_v == 0:
                    tl.store(rp_k + ((slot * Hg + i_h) * L_ring + w) * K + o_k,
                             b_k.to(rp_k.dtype.element_ty), mask=mask_k)
                    tl.store(rp_g + (slot * H + i_hv) * L_ring + w, b_g)
                w += 1

    if will_flush:
        # Fold everything into the checkpoint and empty the ring.
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)
    # rp_pos is per-request and SHARED across layers, so exactly one program may
    # write it -- every layer computes the same value from the same inputs.
    if i_v == 0 and i_hv == 0 and i_l == 0:
        tl.store(rp_pos + i_n, tl.where(will_flush, 0, w))


def append_ssm_inputs_along_accept_paths(
    k_stash: torch.Tensor,
    v_stash: torch.Tensor,
    g_stash: torch.Tensor,
    beta_stash: torch.Tensor,
    ssm_states: torch.Tensor,
    replay_d: torch.Tensor,
    replay_k: torch.Tensor,
    replay_g: torch.Tensor,
    replay_pos: torch.Tensor,
    cache_indices: torch.Tensor,
    ancestor_masks: torch.Tensor,
    last_correct_steps: torch.Tensor,
    T: int,
    Hg: int,
    K: int,
    V: int,
    spec_window: int,
):
    """ReplaySSM commit: append accepted inputs to the ring, flush when needed.

    Drop-in for `advance_ssm_states_along_accept_paths`. `spec_window` is the
    paper's T in the `h + 2T > L` rule -- pass the draft token count so a step that
    accepts a whole window still has room.
    """
    L = ssm_states.shape[0]
    H = ssm_states.shape[2]
    B = cache_indices.shape[0]
    assert ancestor_masks.dim() == 3, "ancestor_masks must be [B, BT, BT // 64]"
    BT, NW = ancestor_masks.shape[-2], ancestor_masks.shape[-1]
    assert NW == BT // 64 and T <= BT, (BT, NW, T)
    L_ring = replay_g.shape[-1]
    # Accept either a single layer's view or the full per-layer tensor; the grid
    # covers layers, so the full tensor is what the backend should pass.
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (L, triton.cdiv(V, BV), B * H)
    tree_verify_state_append_kernel[grid](
        k_stash=k_stash,
        v_stash=v_stash,
        g_stash=g_stash,
        beta_stash=beta_stash,
        ssm_states=ssm_states,
        rp_d=replay_d,
        rp_k=replay_k,
        rp_g=replay_g,
        rp_pos=replay_pos,
        cache_indices=cache_indices,
        ancestor_masks=ancestor_masks,
        last_correct_steps=last_correct_steps,
        T=T,
        L_ring=L_ring,
        stride_kl=k_stash.stride(0),
        stride_vl=v_stash.stride(0),
        stride_gl=g_stash.stride(0),
        stride_bl=beta_stash.stride(0),
        stride_sl=ssm_states.stride(0),
        stride_rdl=replay_d.stride(0) if replay_d.dim() == 5 else 0,
        stride_rkl=replay_k.stride(0) if replay_k.dim() == 5 else 0,
        stride_rgl=replay_g.stride(0) if replay_g.dim() == 4 else 0,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        NW=NW,
        BK=BK,
        BV=BV,
        SPEC_WINDOW=int(spec_window),
    )


def advance_ssm_states_along_accept_paths(
    k_stash: torch.Tensor,
    v_stash: torch.Tensor,
    g_stash: torch.Tensor,
    beta_stash: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    ancestor_masks: torch.Tensor,
    last_correct_steps: torch.Tensor,
    T: int,
    Hg: int,
    K: int,
    V: int,
    track_steps: torch.Tensor = None,
    track_slots: torch.Tensor = None,
):
    """Advance pooled SSM states along each request's accept path.

    Stashes are [L, S, T, ...] flattened on the last dims; only the first
    len(cache_indices) stash slots are read. ssm_states is the full
    [L, pool, H, V, K] pool tensor; states are updated in place.

    track_steps/track_slots optionally snapshot the state right after a given
    accepted tree slot into another pool slot (mamba prefix-cache tracking).
    """
    L = ssm_states.shape[0]
    H = ssm_states.shape[2]
    B = cache_indices.shape[0]
    assert ancestor_masks.dim() == 3, "ancestor_masks must be [B, BT, BT // 64]"
    BT, NW = ancestor_masks.shape[-2], ancestor_masks.shape[-1]
    assert NW == BT // 64 and T <= BT, (BT, NW, T)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (L, triton.cdiv(V, BV), B * H)
    tree_verify_state_advance_kernel[grid](
        k_stash=k_stash,
        v_stash=v_stash,
        g_stash=g_stash,
        beta_stash=beta_stash,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        ancestor_masks=ancestor_masks,
        last_correct_steps=last_correct_steps,
        track_steps=track_steps,
        track_slots=track_slots,
        T=T,
        stride_kl=k_stash.stride(0),
        stride_vl=v_stash.stride(0),
        stride_gl=g_stash.stride(0),
        stride_bl=beta_stash.stride(0),
        stride_sl=ssm_states.stride(0),
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        NW=NW,
        BK=BK,
        BV=BV,
        HAS_TRACK=track_steps is not None,
        num_warps=1,
        num_stages=2,
    )


def chunk_gated_delta_rule_tree_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ancestor_masks: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Block-quadratic tree verification for the gated delta rule.

    Args:
        q, k: `[B, T, Hg, K]`, v: `[B, T, H, V]`, g, beta: `[B, T, H]`
            (g is the raw per-token log decay; the path cumsum is internal).
        ancestor_masks: `[B, TREE_CHUNK_SIZE, 1]` int64 from
            build_tree_ancestor_masks (2D `[B, TREE_CHUNK_SIZE]` also accepted).
        initial_state_source: `[pool, H, V, K]` per-layer state pool (read-only here).
        initial_state_indices: `[B]` state slot per request.

    Returns:
        o: `[B, T, H, V]` verification outputs. No state is written.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = TREE_CHUNK_SIZE
    assert T <= BT, f"chunk tree verify supports at most {BT} draft tokens, got {T}"
    if ancestor_masks.dim() == 3:
        # Multi-word masks exist for the fused verify path only; these kernels
        # index single int64 words.
        assert ancestor_masks.shape[-1] == 1 and ancestor_masks.shape[-2] == BT
        ancestor_masks = ancestor_masks[..., 0]
    assert q.dtype != torch.float32, "use bfloat16/float16 inputs"

    if scale is None:
        scale = K**-0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # The kernels assume densely packed [B, T, H, D]; q/k/v may arrive as
    # non-contiguous views of the fused qkv projection.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    g_path = torch.empty_like(g, dtype=torch.float32)
    tree_path_cumsum_kernel[(B,)](
        g=g,
        g_path=g_path,
        ancestor_masks=ancestor_masks,
        T=T,
        H=H,
        BT=BT,
        BH=max(16, triton.next_power_of_2(H)),
    )

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    tree_kkt_solve_kernel[(B * H,)](
        k=k,
        g=g_path,
        beta=beta,
        ancestor_masks=ancestor_masks,
        A=A,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BC=16,
    )

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g_path,
        cu_seqlens=None,
    )

    o = torch.empty_like(v)
    grid = (triton.cdiv(V, 64), B * H)
    tree_verify_o_kernel[grid](
        q=q,
        k=k,
        w=w,
        u=u,
        g=g_path,
        ancestor_masks=ancestor_masks,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        o=o,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
        num_warps=4,
        # num_stages=1: the SM100 software pipeliner crashes on this
        # multi-dot K-loop (TritonGPUPipeline MLIR failure); the loop is
        # only K/BK=2 iterations, so pipelining buys nothing anyway.
        num_stages=1,
    )
    return o
