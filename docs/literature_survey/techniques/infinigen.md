# InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  When generating text from long prompts or across large batches, the KV cache grows so massive that it cannot fit inside GPU memory. To prevent Out-Of-Memory crashes, systems offload the cache to cheap, plentiful host CPU RAM. However, repeatedly hauling hundreds of megabytes or gigabytes of KV data back across the slow PCIe bus for every single generated token chokes the GPU, turning high-speed AI chips into idle silicon waiting on data transfers. Existing cache-pruning methods (like permanently deleting tokens deemed unimportant) fail because attention is dynamic: a fact ignored on token five might suddenly become crucial fifty tokens later.

- **How InfiniGen Solves It (Step-by-Step in Plain Words):**
  1. **Before / During Prefill (Offline Rotation & Channel Slicing):** Before running the model, InfiniGen inspects the query weight matrices offline and rotates them so that the most important numerical information gets concentrated into just a few specific channel columns. During the prompt reading phase, it identifies the top thirty percent of these concentrated channels and keeps only this narrow slice on the GPU as a lightweight prediction proxy. The entire full-precision KV cache is moved to CPU RAM.
  2. **Where the Data Lives (Storage & Compression):** The complete, uncompressed cache of all past tokens sits safely in host CPU memory, meaning no past tokens are permanently erased. The GPU holds only the thirty-percent slice of Key channels (taking just fifteen percent of normal KV memory) plus the model weights. If host CPU RAM ever fills up, a lightweight frequency counter evicts tokens that are rarely recalled.
  3. **During Decode (One-Step-Ahead Rehearsal & Pipelined Fetch):** As the model computes layer one, it peeks ahead to speculate what layer two will need. Because consecutive layers receive nearly identical input signals due to normalization and residual connections, layer one can multiply its current input by the narrow proxy slice to estimate upcoming attention scores. Any token predicted to receive less than one percent of attention weight is skipped. While layer one finishes its heavy neural network calculations, the GPU uses high-speed background direct memory access to pull only the vital five to ten percent of tokens from CPU RAM over PCIe. When layer two begins, its required data is already sitting in GPU memory, completely avoiding transfer stalls.

- **How It Compares to Our Project:**
  InfiniGen is a premier example of **Spatial** (offloading full cache to CPU RAM) and **Temporal** (speculative prefetching one layer ahead) co-optimization. While InfiniGen relies on a dynamic prefetch threshold to bring uncompressed tokens across PCIe on demand, our project pairs priority scoring with **Tiered Quantization** (compressing secondary tokens to 4-bit) so that even more context stays resident in GPU VRAM and transfers are further minimized.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Up to three times faster inference throughput than competing offload systems; up to thirty-two percentage point higher accuracy than permanent eviction methods (matching full uncompressed cache perplexity); reduces PCIe transfer volume by ten-fold to twenty-fold, hiding nearly all transfer latency behind GPU computation.
  - **What You Pay:** Requires storing a thirty-percent partial Key cache slice in GPU VRAM (adding about fifteen percent memory overhead over an empty cache); requires custom asynchronous transfer pipelining between adjacent transformer layers; does not compress tokens in host RAM (they remain full precision).

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Consecutive Layer Input Invariance:** Transformer block inputs between adjacent layers share an astonishingly high cosine similarity (eighty-nine to ninety-seven percent across OPT and LLaMA models). This occurs because layer normalization shrinks the magnitude of layer updates while residual connections preserve heavy outlier channels. We can exploit this property to compute priority scores or speculative prefetch decisions for layer *N* using the activations already sitting in the registers of layer *N minus one*, eliminating extra round trips.
  2. **Mathematical Equivalence via Orthogonal Rotation:** By multiplying both the query and key projection weights by an orthogonal matrix derived from Singular Value Decomposition, the dot product between query and key remains mathematically identical with zero loss in output quality. However, the energy is concentrated into a few outlier columns, allowing a thirty-percent channel slice to accurately predict attention scores.
  3. **Exponential Cutoff Rule for Softmax Sparsity:** Because softmax exponentiates attention scores, subtracting four or five from the maximum score means that any token below this threshold contributes less than one percent to the final attention output. Setting a relative threshold (maximum minus alpha) automatically adapts to prompt length and prevents transferring useless tokens without needing a hardcoded top-k count.
  4. **Counter-Based CPU Memory Pool Eviction:** When CPU RAM is bounded, keeping a simple access count per token and halving all counters when any counter approaches saturation provides identical perplexity to Least-Recently-Used eviction while eliminating lock-synchronized linked list overhead.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Irregular Token Gathering Over PCIe:** InfiniGen prefetches arbitrary, non-contiguous token indices at every decoding step. On consumer PCIe (such as Gen 4 x8 on an RTX 4060), scattered direct memory access transfers of individual token vectors suffer severe bandwidth degradation compared to transferring contiguous memory chunks or pages.
  - **Evaluated on Workstation GPUs:** The paper evaluated its runtime on an NVIDIA RTX A6000 with 48 GB VRAM. On an 8 GB RTX 4060 or a 16 GB RTX 4090, allocating fifteen percent of VRAM to partial Key proxies while fitting a 3B or 4B BF16 model leaves very narrow headroom for batching.
  - **No Host-Side Quantization:** InfiniGen keeps the entire CPU cache pool in uncompressed sixteen-bit precision. On consumer rigs with sixteen gigabytes of system RAM, a long multi-turn session across several concurrent requests will rapidly exhaust host RAM unless paired with four-bit host compression.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our manual notes correctly identified the core mechanics: storing a small proxy signal on GPU, offloading the bulk cache to host RAM, and prefetching over PCIe ahead of the computation cycle to prevent stalls.
  - *Clarification from PDF:* The team note listed models as LLaMA-2 7B, 13B, and 70B. In reality, the authors evaluated OPT (6.7B, 13B, 30B), LLaMA-2 (7B, 13B), LLaMA-2-7B-32K, and LLaMA-3-8B-1048K; 70B was not tested. Furthermore, the "proxy" is not an auxiliary embedding model, but rather a mathematically rotated thirty-percent slice of the model's own Query and Key weights.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Wonbeom Lee*, Jungi Lee*, Junghwan Seo, Jaewoong Sim (* equal contribution) — Seoul National University
- **Publication / Venue & Year:** 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24), July 2024. (arXiv:2406.19707v1, June 2024)
- **Primary Axis:** Spatial (Host CPU memory offloading with selective PCIe prefetching)
- **Secondary Interacting Axis:** Temporal (One-layer-ahead speculative attention rehearsal) & Structural (Dynamic sparse token selection and counter-based host cache pruning)
- **Official Codebase / Repository:** [https://github.com/snu-comparch/infinigen](https://github.com/snu-comparch/infinigen)
- **Baseline Implementation Stack:** Built on top of FlexGen and CUDA Unified Virtual Memory (UVM) runtimes in PyTorch and CUDA C++. Evaluated across OPT (6.7B, 13B, 30B), LLaMA-2 (7B, 13B), LLaMA-2-7B-32K, and LLaMA-3-8B-1048K.

---

## 3. How the Algorithm Actually Works (Under the Hood)

```
====================================================================================================
                             INFINIGEN ARCHITECTURE & TIMING PIPELINE
====================================================================================================

OFFLINE:
  Weight Matrix W_Q  ──> [SVD: U, Sigma, V^T] ──> Skew Matrix A = V
  Rotated Weights:   W'_Q = W_Q * A ,  W'_K = W_K * A  (Zero loss: (Q*A)(K*A)^T = Q*K^T)

PREFILL STAGE:
  Prompt Tokens      ──> Top 30% Outlier Column Channels Selection (Column sum of |Q'| + |K'|)
  GPU VRAM:          Holds W_Q^partial (2.5% param size) and K^partial (15% KV size)
  CPU Host RAM:      Holds 100% full-precision KV Cache Pool

DECODE STAGE (Token Generation Pipeline):
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ LAYER (i - 1):                                                                                  │
 │   Attention Computation                                FFN Computation                          │
 │  [═════════════════════]                             [═════════════════════════════════════════]│
 │                                                                                                 │
 │   Concurrently (Speculation):                         Concurrently (Overlapped PCIe DMA):       │
 │   Q_hat_i = X_a^(i-1) * W_Q,i^partial                 Prefetch vital Keys & Values for Layer i  │
 │   Score_i = Q_hat_i * (K_i^partial)^T                 Tokens where Score_i >= max(Score) - alpha│
 │  [═════════════════════]                             [═════════════════════════════════════════]│
 └───────────────────────────────────────────────────────────────────┬─────────────────────────────┘
                                                                     │ DMA completes
 ┌───────────────────────────────────────────────────────────────────▼─────────────────────────────┐
 │ LAYER i:                                                                                        │
 │   Light Attention (Only prefetched 5-10% tokens)        FFN Computation                         │
 │  [═════════════════════]                             [═════════════════════════════════════════]│
 │                                                                                                 │
 │   Concurrently (Speculation for Layer i+1):           Concurrently (Overlapped PCIe DMA):       │
 │   Speculate Layer i+1 attention entries               Prefetch Layer i+1 Keys & Values          │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
====================================================================================================
```

- **Data Structures Added / Modified:**
  - **Skewed Weight Matrices ($W'_{Q, \ell}, W'_{K, \ell}$):** Statically rotated offline weight matrices where $W'_{Q, \ell} = W_{Q, \ell} A_\ell$ and $W'_{K, \ell} = W_{K, \ell} A_\ell$. Matrix dimensions remain identical to native model dimensions ($D \times D$).
  - **Orthogonal Transformation Matrix ($A_\ell$):** An orthogonal matrix $A_\ell = V_\ell \in \mathbb{R}^{d \times d}$ derived from Singular Value Decomposition of the query activation matrix.
  - **Partial Query Weight ($W_{Q, \ell}^{\text{partial}}$):** A sub-matrix of size $D \times k$ containing only the top $k = 0.30 \cdot D$ outlier channels, resident on GPU VRAM (overhead $\approx 2.5\%$ of model size).
  - **Partial Key Cache ($K_\ell^{\text{partial}}$):** A sliced Key cache of size $N \times k$ storing the top thirty percent channels of past tokens on GPU VRAM (overhead $\approx 15\%$ of full KV cache footprint).
  - **Host KV Cache Pool:** A contiguous buffer in host CPU RAM storing full-precision FP16 Key and Value tensors for all $N$ tokens across all layers.
  - **Token Access Frequency Counters:** A flat integer counter array in host CPU memory tracking the retrieval count of each token for victim eviction under memory pressure.

- **Scoring, Compression & Decision Logic:**
  1. *Mathematical Equivalence of Weight Skewing:*
     $$\tilde{Q} = X_a W_Q A, \quad \tilde{K} = X_a W_K A$$
     $$\tilde{Q} \tilde{K}^T = (X_a W_Q A)(X_a W_K A)^T = X_a W_Q A A^T W_K^T X_a^T = X_a W_Q W_K^T X_a^T = Q K^T$$
     *Plain English Translation:* Rotating both the Query and Key projection matrices with the exact same orthogonal matrix $A$ preserves the attention inner product identically because an orthogonal matrix transposed is its own inverse ($A A^T = I$). This introduces zero numerical degradation while aligning matrix variance with standard coordinate axes.
  2. *SVD Alignment Formula:*
     $$Q = U \Sigma V^T \implies A = V \implies \tilde{Q} = Q A = U \Sigma V^T V = U \Sigma$$
     *Plain English Translation:* Singular Value Decomposition splits the query matrix into rotation matrices $U, V$ and singular value scale matrix $\Sigma$. Setting the transformation matrix $A$ equal to $V$ rotates the query vectors so that their largest singular values align directly along the first few coordinate columns.
  3. *Outlier Column Index Selection (Prefill Phase):*
     $$C = \sum_{r=1}^N \left( |\tilde{Q}_{r, :}| + |\tilde{K}_{r, :}| \right), \quad \mathcal{I} = \text{Top-}k(C, \, k = 0.30 \cdot D)$$
     *Plain English Translation:* To find which channels dominate attention dot products, InfiniGen adds the absolute values of the skewed Query and Key matrices across all prompt tokens, calculates the column-wise sum vector $C$, and extracts the top thirty percent column indices $\mathcal{I}$.
  4. *Speculative Attention Rehearsal (Decoding Phase):*
     $$\hat{Q}_i^{\text{partial}} = X_a^{(i-1)} W_{Q, i}^{\text{partial}}$$
     $$\hat{S}_i = \hat{Q}_i^{\text{partial}} \left( K_i^{\text{partial}} \right)^T$$
     *Plain English Translation:* At decoding step $t$, layer $i-1$ takes its current normalized input $X_a^{(i-1)}$ (which has an average cosine similarity of $0.90\text{--}0.97$ to layer $i$'s future input) and multiplies it by layer $i$'s partial query weights to create an estimated query $\hat{Q}_i^{\text{partial}}$. Multiplying this estimated query by the partial key cache already on the GPU generates the speculated attention score vector $\hat{S}_i$ for layer $i$.
  5. *Dynamic Softmax Thresholding:*
     $$\tau_i = \max(\hat{S}_i) - \alpha, \quad \text{Fetch token } j \iff \hat{S}_i(j) \ge \tau_i$$
     *Plain English Translation:* InfiniGen identifies the maximum speculated score across all past tokens and sets the cutoff threshold $\tau_i$ by subtracting $\alpha$ (empirically set to $4$ for OPT and $5$ for LLaMA-2). Because attention weights scale exponentially after softmax ($\exp(\max - \alpha) / \exp(\max) = \exp(-\alpha)$), any token below this cutoff receives under $0.7\%$ to $1.8\%$ of attention weight and can be safely omitted.
  6. *Host Cache Pool Eviction with Saturated Aging:*
     $$\text{Victim} = \arg\min_{j \in [1, N]} c_j; \quad \text{if } \max_j c_j \ge C_{\max}, \; c_j \leftarrow \lfloor c_j / 2 \rfloor$$
     *Plain English Translation:* When host CPU RAM exceeds its capacity cap (e.g., 80% of total prompt memory), the token with the lowest access count $c_j$ is evicted and overwritten by the newest token. When any counter reaches the maximum integer limit $C_{\max}$, all counters are divided by two to age out older, inactive tokens.

- **Execution Cadence:**
  - *Offline:* SVD decomposition and weight matrix rotation run once per model architecture.
  - *Prefill:* Sliced column index selection runs once per request prompt.
  - *Decode:* Speculative query projection, score thresholding, and asynchronous PCIe DMA prefetch run at every single generated token, pipelined between adjacent layers.

- **Memory Hierarchy & Placement:**
  - **Keys ($K$):** Complete uncompressed FP16 Key cache resides in host CPU RAM. A thirty-percent channel slice ($K^{\text{partial}}$) resides permanently in GPU VRAM (fifteen percent of the total KV footprint).
  - **Values ($V$):** One hundred percent of Value vectors reside exclusively in host CPU RAM in FP16 precision. Zero Value vectors are held permanently on GPU; only the selected five to ten percent are transferred transiently during prefetch.
  - **Granularity:** Token-level selection across heads. Within a layer, the prefetch count is averaged across heads to ensure uniform tensor dimensions for matrix multiplications, with a strict safety cap of twenty percent maximum tokens per layer.

- **Decode-Phase Overheads:**
  - *Speculative Rehearsal Compute:* Extremely fast ($<0.2\text{ ms}$ on modern GPUs), fully hidden behind the attention computation of layer $i-1$.
  - *PCIe Transfer Overlap:* Because only five to ten percent of KV entries are retrieved, PCIe transfer latency is reduced from $27.1\text{ ms}$ down to $\approx 1.5\text{ ms}$ per block on OPT-13B, completely overlapping with the Feed-Forward Network (FFN) computation of layer $i-1$.
  - *GPU Stalls:* Near zero. Total block execution time is $4.5\text{ ms}$, within $1.52\times$ of the ideal theoretical GPU speed ($3.0\text{ ms}$ with zero offloading overhead).

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

- **Memory Saved (Efficiency):**
  - **GPU Memory Footprint:** The partial query weights occupy only $2.5\%$ of total model parameter memory. The partial key cache occupies only $15.0\%$ of the total KV cache footprint.
  - **Active Cache Load:** Reduces active GPU attention memory to $5\%\text{--}10\%$ of full sequence length on average (less than $73$ tokens loaded out of $2,048$ tokens on OPT-13B).
  - **Host RAM Budgeting:** Evaluated under an $80\%$ CPU memory cap with counter-based eviction, maintaining zero loss in perplexity compared to an unconstrained $100\%$ cache pool.

- **Speed & Latency (Prefill, Decode, Throughput):**
  - **Total Latency on OPT-13B ($1,920$ input, $128$ output tokens, batch size $20$):**
    - CUDA Unified Virtual Memory (UVM): $2,007.4\text{ s}$ (dominated by prefill page faults).
    - UVM + H2O ($20\%$ budget): $1,737.4\text{ s}$.
    - FlexGen (Full FP16 Offload): $\approx 550.0\text{ s}$.
    - FlexGen + INT4 Quantization: $\approx 380.0\text{ s}$.
    - FlexGen + H2O ($20\%$ budget): $\approx 270.0\text{ s}$.
    - **InfiniGen:** **$61.0\text{ s}$** total ($15.0\text{ s}$ prefill + $46.0\text{ s}$ decode), achieving a **$4.43\times$ speedup over H2O**, **$6.23\times$ over INT4**, **$9.02\times$ over FlexGen**, and **$32.93\times$ over UVM**.
  - **Per-Block Execution Breakdown (OPT-13B, batch $8$, length $2,048$):**
    - FlexGen: $28.0\text{ ms}$ total ($96.9\%$ spent waiting on PCIe data transfer).
    - H2O ($20\%$ budget): $\approx 13.0\text{ ms}$ ($91.8\%$ spent on PCIe transfer).
    - INT4 Quantization: $\approx 17.5\text{ ms}$ (PCIe transfer plus dequantization overhead).
    - **InfiniGen:** **$4.5\text{ ms}$** total (data transfer dropped to $\approx 1.5\text{ ms}$, completely overlapped with FFN). Only $1.52\times$ slower than theoretical Ideal ($3.0\text{ ms}$ pure GPU execution).
  - **Throughput Scaling with Batch Size (OPT-13B, batch $4 \to 20$):**
    - InfiniGen throughput expands from $27.36$ to **$41.99\text{ tokens/s}$**.
    - H2O throughput stagnates from $21.31$ to $25.70\text{ tokens/s}$.
    - INT4 throughput stagnates from $12.22$ to $14.02\text{ tokens/s}$.
  - **Tail Latency (P95/P99):** Not reported in paper (mean throughput and total wall-clock execution time only).

- **Output Quality & Trustworthiness:**
  - **Language Modeling Perplexity on WikiText-2 and PTB (Sequence Length $2,048$, Table 2):**

| Model | Setup | WikiText-2 PPL | PTB PPL |
| :--- | :--- | :---: | :---: |
| **OPT-6.7B** | 100% Full Cache | 11.68 | 13.86 |
| | 80% Cap (FIFO) | 19.64 | 16.82 |
| | 80% Cap (LRU) | 11.68 | 13.85 |
| | **80% Cap (Counter)** | **11.68** | **13.86** |
| **OPT-13B** | 100% Full Cache | 10.55 | 12.78 |
| | 80% Cap (FIFO) | 30.99 | 33.84 |
| | **80% Cap (Counter)** | **10.55** | **12.78** |
| **OPT-30B** | 100% Full Cache | 10.14 | 12.31 |
| | 80% Cap (FIFO) | 30.66 | 35.45 |
| | **80% Cap (Counter)** | **10.14** | **12.31** |
| **LLaMA-2-7B** | 100% Full Cache | 5.69 | 22.53 |
| | 80% Cap (FIFO) | 22.26 | 61.88 |
| | **80% Cap (Counter)** | **5.69** | **22.53** |
| **LLaMA-2-13B**| 100% Full Cache | 5.25 | 31.94 |
| | 80% Cap (FIFO) | 21.41 | 32.34 |
| | **80% Cap (Counter)** | **5.25** | **31.94** |

  - **Downstream Few-Shot Accuracy (lm-evaluation-harness 5-shot):**
    - Evaluated across COPA, OpenBookQA, WinoGrande, PIQA, and RTE.
    - At relative cache sizes under $10\%$, InfiniGen achieves up to a **$32.6\text{ percentage point}$ accuracy lead over H2O and INT4**, tracking within $0.5\%$ of the uncompressed full-cache baseline.
  - **Extreme Long Context (LLaMA-2-7B-32K on WikiText-2 up to $32,768$ tokens):**
    - Even when retaining only $64$ active tokens, InfiniGen maintains steady perplexity ($\approx 5.8$), whereas H2O diverges sharply above $12.0\text{ PPL}$ due to catastrophic permanent token eviction.

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Compliant. While InfiniGen performs an offline SVD rotation on $W_Q$ and $W_K$, the weights remain in full BF16/FP16 precision with identical dimensions ($D \times D$). The transformation is strictly an orthogonal change of basis ($A A^T = I$) and requires no weight quantization or fine-tuning.
  - **Dense GQA Compatibility:** The paper validated InfiniGen primarily on Multi-Head Attention (OPT) and Grouped-Query Attention (LLaMA-2 / LLaMA-3). For our pinned models (**Qwen 3 4B**, **Phi-4 mini**, **Ministral 3 3B**, **SmolLM3 3B**, **Qwen 3 1.7B**), GQA groups multiple query heads per key-value head (e.g., 4:1 or 8:1 ratio). In GQA, the rehearsal projection $\hat{Q}$ computes dot products against shared Key channels; thresholding $\max(\hat{S}) - \alpha$ is simply pooled across grouped query heads before initiating the prefetch DMA.

- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - **PCIe Bandwidth Constraints:** InfiniGen was benchmarked on server PCIe 3.0 x16 ($\approx 12\text{--}14\text{ GB/s}$ real throughput). A local RTX 4060 runs on PCIe 4.0 x8, providing identical effective bandwidth ($\approx 12.5\text{ GB/s}$). InfiniGen's demonstrated ability to compress per-token transfer volume to $5\%\text{--}10\%$ is mandatory to avoid GPU starvation on consumer boards.
  - **Host RAM Budget:** On a 16 GB system RAM machine, storing uncompressed FP16 KV caches for long sequences ($32\text{K}$) limits concurrent batch capacity. For our project, pairing InfiniGen's prefetch mechanism with **host-side 4-bit compression** is essential to prevent host RAM exhaustion.
  - **VRAM Budget:** For Qwen 3 4B (model weights $\approx 8\text{ GB}$ in BF16), allocating $2.5\%$ for partial query weights ($\approx 200\text{ MB}$) and $15\%$ for partial key cache at $8\text{K}$ context ($\approx 150\text{ MB}$) fits comfortably within both the 8 GB and 16 GB VRAM budgets.

- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** **High**. Requires asynchronous CUDA stream synchronization overlapping with PyTorch forward hooks, custom DMA gather operations for non-contiguous token indices, and specialized attention kernels.
  - **WP-3 (Micro-Benchmarking):** We can directly implement InfiniGen's SVD rotation and speculative rehearsal in our PyTorch benchmarking harness to profile the exact correlation between layer $i-1$ input embeddings and layer $i$ attention patterns across our pinned models.
  - **WP-6 Engine Integration (`vLLM` PagedAttention / `LMCache`):**
    - *The Paged Block Fragmentation Hurdle:* Standard vLLM allocates KV cache in fixed 16-token virtual pages. InfiniGen selects arbitrary, fine-grained individual token indices, which causes severe memory fragmentation if mapped directly into PagedAttention.
    - *Adaptation for Our System:* Rather than fetching individual token vectors, we can adapt InfiniGen to perform **Block-Level Speculative Prefetching**. If any token in a 16-token page exceeds the threshold $\max - \alpha$, the entire 16-token block is scheduled for DMA retrieval. This allows seamless integration with LMCache's chunk-level tiered storage and vLLM's page tables.

- **Classification for Project:** **Component Inspiration for WP-4 & Comparison Baseline for WP-6** (Serves as our primary architectural blueprint for overlapping CPU-to-GPU prefetching with transformer compute, and a strong baseline against our static priority scoring).
