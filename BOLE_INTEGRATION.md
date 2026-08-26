# Bole integration — full scope and preparation

Paper: *Bole: Efficient Tree Speculation for Hybrid-Attention Language Models*,
arXiv 2608.01651.

---

## The objective: recover CUDA-graph capture memory

**This is the goal, stated first because it is what the kernel work is for.**

Graph capture memory, measured from real server logs
(`Capture target verify CUDA graph end ... mem usage=`):

| config | draft tokens | captured batch sizes | capture GB |
|---|---|---|---|
| c1 b31 | 32 | 1 | 0.19 |
| c8 b31 | 32 | 8 | 1.14 |
| c16 b31 | 32 | 16 | 2.15 |
| c32 b31 | 32 | 32 | 4.61 |
| c32 b15 | 16 | 32 | 2.04 |
| c32 b17 | 18 | 32 | 2.78 |
| c32 b63 | 64 | 32 | 8.86 |

Linear in both dimensions: **~4.5 MB per (batch size x draft token)**.

That memory is not free real estate — it is *lost* memory:

* **c64 is unservable at budget 31.** Runtime OOMs at every fraction tried
  (0.75 / 0.72 / 0.70 / 0.68); the failing allocations are 0.7-1.3 GiB of
  activations for 2048 verify tokens.
* **Graphs cannot simply be dropped.** Measured at c32 budget 17: cutting
  `--cuda-graph-max-bs` from 32 to 1 saves 2.22 GB and costs **12% throughput**
  (2998.7 -> 2639.5 tok/s). Graphs earn their memory; the problem is the bill.
* **Every GB reclaimed is KV cache**, i.e. more concurrent requests at the same
  hardware — which is the Regime B objective.

**Why a fused kernel reclaims it.** Capture memory scales with draft tokens, which
means it is dominated by per-token verify intermediates held live across launches.
The current paths keep those in HBM between stages:

| path | launches per layer | live intermediates between launches |
|---|---|---|
| plain (`chunk_tree_verify.py`, T <= 64) | 3 — `tree_path_cumsum`, `tree_kkt_solve`, `tree_verify_o` | path cumsum, KKT factors |
| fused (`gdn_tree_triton.py`, T > 64) | 4 — K0 scalars, K1 gram, K2 solve, K3 out | `prefix`, `KK`, `QK`, `D`, `D_strict`, `A`, `Ainv`, `U` |

x48 GDN layers = **144 or 192 launches per verify step.**

Bole's kernel holds the equivalents in registers for the lifetime of one CTA:
`B_0^(r)`, `R^(r)`, `Z^(m)` **never written to HBM**; only `O^(r)` and the commit
factors `U^(r)` leave. Fewer live intermediates is directly fewer bytes the graph's
private pool must reserve.

**This is the claim to verify, and it is not yet measured.** The mechanism is
plausible from the code, but "fused kernel reduces capture memory by X" must be
measured end to end (server log capture line), not asserted, and not inferred from
a microbenchmark. An earlier microbenchmark on the fused path was wrong by 2.7x.

---

## Component status

| Bole component | Ours | Status |
|---|---|---|
| Parallel tree verification (vs serial recurrence, their L1) | chunk/fused Triton kernels | **have** — their 3.4-7.7x is vs serial, which we already replaced (1.50x) |
| Factorized state (their L2, 82-99x) | `TreeVerifyStash` + `fold_accepted` | **have** |
| Batch-wide budget `B_ver(c)` | per-request budget, no total cap | **partial** — startup schedule landed (`a3ad480`) |
| Runtime per-request allocation by draft probability | uniform per-request | **not built** — needs ragged verification |
| Value-tiled fused kernel, on-chip Neumann | 3-4 launches, HBM intermediates | **not built** — this document's objective |

---

## Work item 1: fused verify kernel (the memory objective)

**Target the plain path first.** `TREE_CHUNK_SIZE = 64` selects it whenever
`num_draft_tokens <= 64`, and every optimum we have measured at concurrency is
below that (c32 optimum is budget 17 -> T=18). The K0-K3 path only runs at T > 64,
which measures worse at every concurrency tested.

Bole's design, to port:

* **Value-tiled CTAs.** Partition `d_v` into tiles `b_v`; `N_v = ceil(d_v / b_v)`
  independent CTAs, each taking one value tile across the full T-node extent. The
  tree is never partitioned across CTAs, preserving intra-tree parallelism.
  *Note our K2 grid `(B*HV, NV)` already has this shape.*
* **On-chip finite-Neumann.** `Z^(m+1) = -G Z^(m)` for `m < d`, accumulating into
  `U^(r)`, held in registers. Exact by their Lemma 3 (`G^(d+1) = 0`, d = max tree
  depth).
* **Operand lifetimes.** `S_pre` released after the state products; `P, beta`
  discharged after forming `B_0` and `R`; `G` live only during the d propagation
  rounds; `C` loaded for the final readout.
* **Working set per CTA:** `Theta(T~^2 + T~ b_v + b_k(T~ + b_v))`.

**A prior negative result to respect, and why it may not apply.** The existing
kernel's docstring records that merging K3 into K2 measured *worse*: "the output
stage loses its independent grid and serializes behind the solve". That was at the
production config T=129, where `NB = ceil(T/32) = 5`, so K3's extra grid dimension
carries real parallelism. At T=18, `NB = 1` and that dimension is degenerate — so
the measured objection does not obviously transfer to the small trees we now know
are optimal at concurrency. **Unverified: whether NB=1 removes the serialization
penalty.** Measure before building the full port.

**Depth matters and we sit badly on it.** Bole caps tree depth at 8; ours runs to
15 (K=15). The Neumann polynomial needs `d` propagation rounds on-chip, so our
trees need ~2x their rounds at equal node count. Their paper does not isolate depth
sensitivity (no wide-shallow vs narrow-deep at fixed N), so this is ours to
measure.

## Work item 2: batch-wide budget, remaining half

Landed: `SGLANG_DFLASH_TFM_BUDGET_SCHEDULE` picks the budget at startup from
`max_running_requests` (`a3ad480`).

Not landed, and blocked:

* **Per-step capacity selection.** `draft_token_num` is baked into graph capture,
  so varying it per step needs a captured graph per capacity. Cost is now known —
  ~4.5 MB per (batch x token), so six families are 7.0 GB at c8, 27.8 GB at c32,
  55.7 GB at c64. Affordable below c64, and cheaper still once work item 1 shrinks
  the per-slot cost.
* **Non-uniform allocation across requests.** Bole scores nodes by path probability
  `rho(v) = prod p_draft(u | parent(u))`, takes a batch-global top-N, and relies on
  ancestor probabilities upper-bounding descendants so prefix-closure is automatic
  — the same monotonicity our builder already has (`score(child) = score(parent) +
  logprob`, `logprob <= 0`).
  **Blocked:** our tensors are `[bs, num_nodes]` and the target verifies
  `bs x draft_token_num` tokens regardless of `node_mask`, so a non-uniform split
  changes *which* nodes each request gets without changing cost. Realising the
  saving needs ragged verification, which is a change to target-side batching.
  `tools/` has a probe (`SGLANG_DFLASH_TFM_DCUT_PROBE`) that reports what a
  batch-global top-N *would* have allocated versus the uniform split; run it before
  building, because our benchmark runs one dataset per batch and homogeneous
  requests are the worst case for non-uniform allocation.

## Work item 3: offline calibration

`B_ver(c) = max{N in G : T_ver(N|c) <= (1+eps) T_dec(c)}`, eps ~ 0.2-0.5, profiled
per bucket over model, GPU, parallelism, **batch size**, **KV length**, and tree
depth/template.

We have the batch dimension (measured optima: budget 95 at c1, 17 at c32) and
**nothing on KV length**, which is one of their three bucketing axes and a dimension
our benchmark never varies. That gap is worth closing before calibration is
automated.

---

## Sequencing

1. **Measure whether NB=1 removes the K3-into-K2 serialization penalty.** Cheap,
   and it decides whether the port's central assumption holds at our tree sizes.
2. **Port the plain path to a single fused kernel**, then measure capture memory
   from the server log — the objective.
3. **Re-run the c64 wall** with the reclaimed memory: does budget 31 fit, and does
   a larger tree beat budget 16 there?
4. **Then** per-step capacity selection, which the reclaimed memory makes cheaper.
5. Ragged verification for non-uniform allocation, gated on the D-cut probe showing
   the spread is worth it.
