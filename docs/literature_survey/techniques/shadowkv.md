# ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  When serving modern large language models on extremely long prompts (up to one million tokens), the key-value memory cache explodes. This memory bloat restricts the server to tiny batch sizes (often just one or two requests at a time), causing severe Out-Of-Memory crashes and terrible serving throughput. Existing solutions suffer from major drawbacks:
  1. *Dynamic sparse attention* selects only important tokens during generation to speed up math, but it keeps the entire cache sitting in expensive GPU memory, failing to solve the memory shortage.
  2. *Naive offloading* moves the cache to cheaper host computer memory, but fetching all those tokens back across the slow connection bus during every single generated token creates massive delays and cripples generation speed.
  3. *Permanent eviction* permanently deletes tokens to cap memory, but permanently discarded context causes severe accuracy collapse when handling multi-turn conversations or questions that reference earlier details.

- **How ShadowKV Solves It (Step-by-Step in Plain Words):**
  1. **Before / During Prefill (The Asymmetry Discovery & Setup):**
     The authors discover a key mathematical asymmetry: before applying rotational position encoding, key vectors naturally collapse into a very small, compact mathematical subspace (they are low-rank), whereas value vectors do not. During prompt reading, ShadowKV compresses these pre-position keys into tiny projection matrices (slashing their size by more than six times). It then chops the post-position keys into short eight-token chunks and computes the average key vector of each chunk to serve as a navigation "landmark". It flags the tiny handful (around 0.2% to 0.3%) of chunks with unusual or dense information (including the very first prompt tokens) as "outliers" and keeps them untouched in full quality on the GPU.
  2. **Where the Data Lives (Storage & Split):**
     ShadowKV splits the cache by tensor type:
     - *On the GPU:* It keeps only the compressed low-rank key projections, the tiny landmark summaries, and the small set of full-quality outlier chunks.
     - *In Host Computer Memory (CPU RAM):* It moves all the large, uncompressed value vectors out of GPU memory entirely.
  3. **During Decode (Generating New Tokens):**
     For every newly generated token, the GPU first uses the lightweight landmarks to quickly score and pick only the tiny slice of chunks (around 1.5% of the total prompt) that actually matter for that query. Then, using two parallel processing streams, the system pulls only those required value chunks from computer memory across the connection bus while simultaneously reconstructing the corresponding full keys from the compact projections directly on the GPU. Because key reconstruction finishes while values are still in transit, key recovery adds zero extra elapsed time. Finally, it checks what was used in the previous step: since back-to-back tokens share roughly 60% of the same chunks, it only fetches and rebuilds the 40% that are genuinely new.

- **How It Compares to Our Project:**
  ShadowKV and our project both tackle the memory wall by combining compression and offloading to preserve complete context without permanent token eviction. However, ShadowKV enforces a rigid tensor-type split—keys always stay on GPU in compressed low-rank form, while values always get offloaded to host memory—and relies on dynamic top-k chunk retrieval. In contrast, our 3-axis project uses a unified priority scoring policy across both keys and values, dynamically keeping high-priority tokens on GPU at full BF16 precision, selectively quantizing medium-priority tokens to 4-bit, and staging cold tokens into host RAM via tiered storage.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Up to 6x larger batch sizes and 3.04x higher serving throughput on an A100 GPU; slashes GPU memory footprint by 6x to 7x; achieves 100% accuracy retention across 1M-token retrieval, RULER, and multi-turn conversations; completely avoids multi-turn amnesia because unselected values remain safe in CPU memory.
  - **What You Pay:** Reading the prompt requires computing an online singular value decomposition per sequence; generating tokens requires managing parallel GPU streams and connection bus transfers; on consumer GPUs with narrow connection lanes, fetching values over PCIe can bottleneck token generation unless batch size and sparsity budgets are carefully balanced.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Key-Value Rank Asymmetry as an Architectural Truth:** The authors prove that pre-position keys can be compressed by over 6x via low-rank decomposition with virtually zero accuracy loss, while compressing values triggers steep accuracy degradation. In our project's selective quantization policy, we can exploit this asymmetry: keys can tolerate aggressive dimensional reduction or lower bit-widths, while values require higher preservation fidelity or tiered offloading.
  2. **Temporal Locality Across Decoding Steps (60% Chunk Reuse):** Adjacent token generation steps share roughly 60% of the exact same top-scoring chunks. Maintaining a small on-GPU working set cache prevents thrashing the connection bus, fetching only the 40% novel chunks per step. This mechanism can be directly incorporated into our temporal scheduling engine.
  3. **Hiding Compute Behind Bus Transfers (Dual-Stream Pipelining):** Reconstructing keys from low-rank matrices takes about 1.2 to 1.5 milliseconds, while transferring values across the bus takes 1.6 to 1.8 milliseconds. By running these concurrently on separate GPU execution streams, key recovery is completely hidden behind the value transfer.
  4. **Outlier Chunk Safeguard (0.2% Static VRAM Anchor):** Grouping tokens into 8-token chunks and keeping the lowest-similarity 0.2% chunks (including the attention sink at token 0) in full precision on GPU prevents accuracy collapse without bloating memory.
  5. **Low-Rank Projection of Newly Generated Tokens (ShadowKV+):** Rather than letting newly generated tokens accumulate in full precision, they can be projected into the existing prefill low-rank subspace on-the-fly, keeping long output sequences equally memory-efficient.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Evaluated Exclusively on High-End Server Hardware (NVIDIA A100 SXM4):** The paper measures throughput on an A100 with 80 GB of High-Bandwidth Memory (2 TB/s bandwidth) and a wide PCIe Gen 4 x16 bus (31.5 GB/s bidirectional). On our local RTX 4060 (272 GB/s VRAM bandwidth, PCIe Gen 4 x8 = 15.75 GB/s) and 16 GB system RAM, PCIe transfer latency is twice as slow and host memory bandwidth is shared with system processes. Under single-batch or small-batch regimes, PCIe transfer overhead can exceed the time saved by sparse attention unless carefully tuned.
  - **Online SVD Overhead During Prefill:** Performing SVD during prompt prefill adds latency (6.65% overhead on 64K context, though decreasing to under 1% on 512K due to quadratic attention scaling). On smaller consumer contexts (4K–32K), SVD overhead can represent a noticeable penalty on Time-To-First-Token (TTFT) unless offloaded or executed asynchronously.
  - **PagedAttention Fragmentation:** Dynamically selecting sparse chunks of size 8 breaks vLLM's standard 16-token page boundary alignment, requiring either custom memory kernels or padding.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our manual notes (Row 9) accurately recognized the compression + offloading synergy, the key-vs-value split, the 60% temporal chunk reuse across steps, cutting transfer in half by moving values only, and hiding key reconstruction behind value fetch.
  - *Refinements & Corrections from PDF:*
    1. The compression on keys is specifically an online SVD on *pre-RoPE* keys because rotary positional embeddings artificially inflate the mathematical rank of keys, whereas pre-RoPE keys possess a naturally sharp singular value decay.
    2. The decode selection uses chunk *landmarks* (mean of 8-token post-RoPE keys) and relies heavily on *outlier* preservation (0.2–0.3% of chunks) to maintain accuracy.
    3. The paper is NOT a permanent eviction method; it is dynamic sparse attention with CPU backing, which is why it maintains 100% accuracy on multi-turn conversations where SnapKV and StreamingLLM collapse.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Hanshi Sun (ByteDance Seed, Carnegie Mellon University), Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu (ByteDance Seed), Harry Dong, Yuejie Chi, Beidi Chen (Carnegie Mellon University)
- **Publication / Venue & Year:** arXiv:2410.21465v3 [cs.LG], 25 Apr 2025 (originally Oct 2024).
- **Primary Axis:** Spatial (Value cache offloading to host CPU memory + low-rank Key retention on GPU + dynamic landmark-based retrieval)
- **Secondary Interacting Axis:** Structural (Low-rank SVD decomposition of pre-RoPE Keys, landmark chunk-mean summarization, outlier extraction) & Temporal (Step-to-step temporal locality cache saving ~60% of transfers across consecutive decode steps)
- **Official Codebase / Repository:** [`https://github.com/ByteDance-Seed/ShadowKV`](https://github.com/ByteDance-Seed/ShadowKV)
- **Baseline Implementation Stack:** Custom PyTorch framework with dedicated CUDA C++ kernels, FlashAttention-2, FlashInfer, and vLLM fused kernels (LayerNorm). Evaluated on LLaMA-3.1-8B, LLaMA-3-8B-1M, GLM-4-9B-1M, Yi-9B-200K, Phi-3-Mini-128K, Qwen2-7B-128K, and LLaMA-3-70B-1M on NVIDIA A100-80GB SXM4.

---

## 3. How the Algorithm Actually Works (Under the Hood)

- **Data Structures Added / Modified:**
  - **Low-Rank Key Projection Buffers ($A, B$):** For pre-RoPE keys across batch size $b$, sequence length $s$, KV heads $h_{\text{kv}}$, head dimension $d$, and rank $r=160$:
    $$A \in \mathbb{R}^{b \times s \times r}, \quad B \in \mathbb{R}^{b \times h_{\text{kv}} \times r \times d}$$
  - **Chunk Landmark Buffer ($L$):** The mean key vectors of non-outlier chunks of size $c=8$:
    $$L \in \mathbb{R}^{b \times h_{\text{kv}} \times (n_c - o) \times d}, \quad \text{where } n_c = \lceil s / c \rceil$$
  - **Static Outlier Cache ($K^{\text{outlier}}, V^{\text{outlier}}$):** Full-precision post-RoPE key and value tensors for the $o$ outlier chunks (typically $o \in [8, 48]$ chunks, $\approx 0.2\%\text{--}0.3\%$ of tokens) pinned in GPU VRAM:
    $$K^{\text{outlier}}, V^{\text{outlier}} \in \mathbb{R}^{b \times h_{\text{kv}} \times (o \cdot c) \times d}$$
  - **Host Value Memory Bank ($V^{\text{CPU}}$):** Full-precision values for all non-outlier tokens offloaded to pinned host CPU memory:
    $$V^{\text{CPU}} \in \mathbb{R}^{b \times h_{\text{kv}} \times (s - o \cdot c) \times d}$$
  - **Temporal Working Set Cache:** An on-GPU ring/LRU buffer tracking selected chunk indices and caching recently fetched value tensors to exploit inter-step token reuse ($\approx 60\%$ hit rate).

```
                      PREFILL PHASE (Online Ingestion)
                      
   Prompt Tokens ──────► Attention Layer ───► Full Keys (K) & Values (V)
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼ (Pre-RoPE)                                                          ▼ (Post-RoPE)
        Truncated SVD (Rank r=160)                                            Chunking (c=8 tokens)
                 │                                                                     │
        ┌────────┴────────┐                                                   Compute Chunk Centroids
        ▼                 ▼                                                            │
    Left Proj (A)   Right Proj (B)                                            Cosine Sim vs Centroids
   [b x s x r]     [b x h x r x d]                                                     │
   (GPU VRAM)      (GPU VRAM)                                              Extract Outliers (0.2-0.3%)
                                                                                       │
                                                                 ┌─────────────────────┴─────────────────────┐
                                                                 ▼                                           ▼
                                                         Outliers (K & V)                           Landmarks (L) & Non-Outlier V
                                                            (GPU VRAM)                                       │
                                                                                          ┌──────────────────┴──────────────────┐
                                                                                          ▼                                     ▼
                                                                                   Landmarks (L)                           Values (V_CPU)
                                                                                    (GPU VRAM)                               (Host RAM)

-----------------------------------------------------------------------------------------------------------------------------------------

                      DECODE PHASE (Token Generation Step)

   Incoming Query (Q) ──────────────────────────────┐
                                                    ▼
                                          Attention with Landmarks (L)
                                                    │
                                          Max-Pool across GQA Heads
                                                    │
                                          Top-k Chunk Selection (k=256)
                                                    │
                                      Compare with Temporal Cache
                                                    │
                                     ┌──────────────┴──────────────┐
                                     ▼ (Missed Chunks ~40%)        ▼
                              CUDA Stream 1                 CUDA Stream 2
                           Fetch Missed Values          Reconstruct Sparse Keys
                             from Host RAM               K_sparse = RoPE(A_I * B)
                             (over PCIe)                   (on GPU Tensor Cores)
                                     │                             │
                                     └──────────────┬──────────────┘
                                                    ▼ (Synchronize Streams)
                                        Assemble Working KV Set:
                                   [Outliers + Reconstructed K / Fetched V]
                                                    │
                                       Dense FlashAttention Kernel
                                                    │
                                             Generated Token
```

- **Scoring, Compression & Decision Logic:**
  1. *Subspace Similarity Metric (Between & Within Sequences):*
     To evaluate whether low-rank subspaces can be shared, the paper defines the similarity metric between two rank-$r$ projection matrices $H_1, H_2 \in \mathbb{R}^{d \times d}$:
     $$D(H_1, H_2) = \frac{\langle H_1, H_2 \rangle_F}{r} = \frac{\operatorname{Tr}(H_1^\top H_2)}{r} \in [0, 1]$$
     *Plain English:* This measures the overlap between two low-rank mathematical subspaces. If $D=1$, the two representations span the exact same subspace; if $D=0$, they are completely orthogonal. Empirical tests show that across different prompts $D < 0.8$ (cannot use static offline SVD), but within a prompt and its future tokens $D > 0.95$ (online SVD per sequence is highly accurate).
  2. *Prefill Low-Rank Decomposition (Algorithm 1):*
     Pre-RoPE keys $K \in \mathbb{R}^{b \times h_{\text{kv}} \times s \times d}$ undergo truncated SVD to rank $r = 160$:
     $$A, B = \operatorname{SVD}(K), \quad \text{such that } K \approx A B$$
     *Plain English:* Decompose the high-dimensional key cache into a compact token-representation matrix $A$ and a basis matrix $B$, slashing key storage by over $6\times$.
  3. *Chunk Landmark Construction & Outlier Detection (Algorithm 1):*
     Post-RoPE keys $K^{\text{RoPE}}$ are grouped into chunks of size $c=8$. For each chunk $j$, the centroid landmark $C_j$ is calculated as the mean:
     $$C_j = \frac{1}{c} \sum_{i=1}^c K_{(j-1)c + i}^{\text{RoPE}}$$
     *Plain English:* Average the 8 key vectors in each chunk to create a single representative landmark vector.
     Within each chunk, the minimum cosine similarity to the centroid is computed:
     $$s_j = \min_{1 \le i \le c} \frac{K_{(j-1)c + i}^{\text{RoPE}} \cdot C_j}{\|K_{(j-1)c + i}^{\text{RoPE}}\|_2 \|C_j\|_2}$$
     Chunks with the lowest minimum similarity form the outlier index set:
     $$I_{\text{outlier}} = \operatorname{arg\,top\text{-}}k\left(-s_j, o\right)$$
     *Plain English:* If any token in a chunk differs sharply from the chunk average, that chunk cannot be accurately approximated. The system flags the $o$ worst-matching chunks as outliers and keeps them uncompressed on the GPU.
  4. *Dynamic Chunk Scoring & Selection During Decode (Algorithm 2):*
     For incoming query $Q \in \mathbb{R}^{b \times h_q \times 1 \times d}$, approximate attention scores are computed against landmarks $L$:
     $$P = Q L^\top \in \mathbb{R}^{b \times h_q \times 1 \times n_c}, \quad S = \operatorname{Softmax}\left(\frac{P}{\sqrt{d}}\right)$$
     To handle Grouped-Query Attention (GQA), scores are summed across the query sequence dimension and max-pooled across query heads sharing the same KV head:
     $$S_1 = \sum_{\text{dim}=-2} S \in \mathbb{R}^{b \times h_q \times n_c}, \quad S_2 = \max_{\text{kv\_group}}(S_1) \in \mathbb{R}^{b \times h_{\text{kv}} \times n_c}$$
     The top-$k$ chunk indices for each KV head are selected:
     $$I = \operatorname{arg\,top\text{-}}k\left(S_2, k\right)$$
     *Plain English:* Test the query against the chunk landmarks. For GQA, take the maximum interest across all query heads assigned to each KV head, and select the top-$k$ winning chunks (typically $k=256$, representing a 1.56% sparse budget).
  5. *Dual-Stream Overlapped Reconstruction:*
     Using two CUDA streams running in parallel:
     $$\text{Stream 1 (PCIe Transfer): } V^{\text{sparse}} = \operatorname{Gather}(V^{\text{CPU}}, I_{\text{missed}})$$
     $$\text{Stream 2 (GPU Compute): } K^{\text{sparse}} = \operatorname{RoPE}\left(\operatorname{Gather}(A, I_{\text{missed}}) \cdot B\right)$$
     *Plain English:* While the GPU waits for the missed value vectors to arrive from CPU memory over PCIe, it simultaneously multiplies matrices $A$ and $B$ and applies RoPE to reconstruct the exact keys. Because matrix multiplication takes ~1.3 ms and PCIe transfer takes ~1.7 ms, key reconstruction finishes before the transfer completes, incurring zero added latency.
  6. *Theoretical Equivalent Bandwidth Formulation:*
     The effective attention bandwidth achieved by ShadowKV is:
     $$\mathcal{B}_e = \frac{2 S B_{\text{GPU}}}{\frac{S}{C} + 2(K + O)C + (1 - \alpha) K C \frac{B_{\text{GPU}}}{B_{\text{PCIe}}}}$$
     - $S$: Sequence length in tokens.
     - $B_{\text{GPU}}$: Raw GPU memory bandwidth (e.g., $2\text{ TB/s}$ on A100, $272\text{ GB/s}$ on RTX 4060).
     - $B_{\text{PCIe}}$: Unidirectional/bidirectional PCIe transfer bandwidth (e.g., $31.5\text{ GB/s}$ on PCIe Gen 4 x16, $15.75\text{ GB/s}$ on Gen 4 x8).
     - $C$: Chunk size ($C = 8$).
     - $K$: Selected chunk budget ($K = 256$ chunks = 2,048 tokens).
     - $O$: Outlier chunk count ($O = 48$ chunks = 384 tokens).
     - $\alpha$: Inter-step temporal hit rate ($\alpha \approx 0.60$, meaning $1-\alpha \approx 0.40$ chunks are cache misses).
     *Plain English Translation:* The numerator represents the total work required to read the entire full-precision KV cache from GPU memory. The denominator sums the three actual costs: reading chunk landmarks ($S/C$), performing attention on the selected sparse tokens and outliers ($2(K+O)C$), and the PCIe transfer delay for the cache misses converted into equivalent GPU memory cycles via the bandwidth ratio $B_{\text{GPU}} / B_{\text{PCIe}}$.

- **Memory Hierarchy & Placement:**
  - **Keys ($K$):** Stored permanently in GPU VRAM in BF16 as low-rank components ($A \in \mathbb{R}^{s \times r}, B \in \mathbb{R}^{r \times d}$) plus landmark centroids ($L \in \mathbb{R}^{(s/c) \times d}$) and static outlier keys ($K^{\text{outlier}} \in \mathbb{R}^{(o \cdot c) \times d}$).
  - **Values ($V$):** Stored primarily in pinned host CPU RAM in BF16 ($V^{\text{CPU}}$). Only outlier values ($V^{\text{outlier}}$) and the dynamically fetched top-$k$ working set reside in GPU VRAM.
  - **Granularity:** Chunk-level (8 tokens) for selection and transfer; token-level for low-rank key projection; layer-by-layer during execution.

- **Decode-Phase Overheads:**
  - Landmark GEMM + Softmax + Top-k selection takes $\approx 0.77\text{--}0.95\text{ ms}$ per Transformer block.
  - Value fetching takes $\approx 1.66\text{--}1.84\text{ ms}$, within which Key reconstruction ($\approx 1.25\text{--}1.51\text{ ms}$) is completely hidden.
  - Attention computation on the sparse subset takes $\approx 0.18\text{--}0.23\text{ ms}$.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

- **Memory Saved (Efficiency):**
  - **Theoretical Memory Footprint Formulation (Appendix A.2):**
    The paper defines the theoretical GPU memory savings ratio as:
    $$\text{Memory Savings} = \frac{2 S M}{\frac{S M}{C} + 2(K + O)C + S r + r M}$$
    For $M = 1024$ elements ($h_{\text{kv}} \times d$), chunk size $C = 8$, context length $S = 131,072$ (128K), selected budget $K = 256$, outliers $O = 48$, and rank $r = 160$, the paper calculates a **$7.08\times$ memory reduction** (or $6.26\times$ if scaling the sparse token term strictly by element size $M$).
  - **Batch Size Scaling:** On an 80 GB A100 GPU running Llama-3-8B-1M:
    - At 60K context: Maximum batch size increases from **8** (Full Attention) to **48** (ShadowKV), a **$6.0\times$ batch size expansion**.
    - At 122K context: Maximum batch size increases from **4** to **24** ($6.0\times$).
    - At 244K context: Maximum batch size increases from **2** to **12** ($6.0\times$).
    - At 488K context: Full Attention crashes with Out-Of-Memory even at batch size 2; ShadowKV successfully runs at batch size 6.

- **Speed & Latency (Throughput & Breakdowns on A100):**
  - **Generation Throughput (Tokens/s) Across Models (Table 3):**
    - **Llama-3.1-8B (8 KV heads):**
      - At 60K context: Full Attention = 160.93 tok/s (BS=8) $\rightarrow$ ShadowKV = **472.77 tok/s** (BS=48) [**$2.94\times$ speedup**; beats Infinite GPU Memory Full Attention baseline of 273.07 tok/s].
      - At 122K context: Full Attention = 80.78 tok/s (BS=4) $\rightarrow$ ShadowKV = **245.90 tok/s** (BS=24) [**$3.04\times$ speedup**; beats Infinite Memory baseline of 134.30 tok/s].
    - **Llama-3-8B-1M (8 KV heads):**
      - At 60K context: Full Attention = 160.62 tok/s (BS=8) $\rightarrow$ ShadowKV = **455.14 tok/s** (BS=48) [**$2.83\times$ speedup**].
      - At 122K context: Full Attention = 80.77 tok/s (BS=4) $\rightarrow$ ShadowKV = **239.51 tok/s** (BS=24) [**$2.97\times$ speedup**].
      - At 244K context: Full Attention = 40.37 tok/s (BS=2) $\rightarrow$ ShadowKV = **119.01 tok/s** (BS=12) [**$2.95\times$ speedup**].
    - **GLM-4-9B-1M (4 KV heads):**
      - At 60K context: Full Attention = 241.05 tok/s (BS=12) $\rightarrow$ ShadowKV = **615.89 tok/s** (BS=50) [**$2.56\times$ speedup**].
      - At 122K context: Full Attention = 122.67 tok/s (BS=6) $\rightarrow$ ShadowKV = **293.40 tok/s** (BS=25) [**$2.39\times$ speedup**].
    - **Yi-9B-200K (4 KV heads):**
      - At 60K context: Full Attention = 204.81 tok/s (BS=10) $\rightarrow$ ShadowKV = **544.36 tok/s** (BS=42) [**$2.66\times$ speedup**].
      - At 122K context: Full Attention = 101.44 tok/s (BS=5) $\rightarrow$ ShadowKV = **260.03 tok/s** (BS=21) [**$2.56\times$ speedup**].
  - **Latency Breakdown Per Transformer Block (Llama-3-8B-1M, Table 12 & 13):**
    - *Prefill Stage (128K context):* Attention = 721.13 ms, FFN = 193.32 ms, SVD = 26.62 ms, Chunk Reduction = 0.20 ms, Cosine Similarity = 2.77 ms, Top-k = 0.14 ms, Gather = 0.02 ms. Total SVD overhead is only **3.25%** of prefill time (drops to **0.97%** at 512K context).
    - *Decode Stage (24 batch $\times$ 128K context):* GEMM + Softmax = 0.58 ms, Max-pool = 0.07 ms, Top-k = 0.15 ms. Value fetch from CPU = **1.66 ms**. Recompute Key on GPU = **1.36 ms** (completely hidden behind Value fetch). Attention compute = 0.21 ms, FFN = 0.29 ms, QKV projection = 0.05 ms.
  - **Efficiency Comparison with Quest Under Extreme Memory Pressure (Table 14):**
    - At $3 \times 1\text{M}$ context on an A100, both Full Attention and Quest run Out-Of-Memory and require host CPU offloading:
      - Full Attention (CPU): 0.21 tokens/s.
      - Quest (CPU offload of both K and V): 9.34 tokens/s.
      - ShadowKV (Values offloaded, low-rank Keys on GPU): **45.32 tokens/s** (**$4.85\times$ faster than Quest**, **$215.8\times$ faster than Full Attention**).
  - **Tail Latency (P95/P99):** Not reported in paper (mean generation throughput and per-block execution times reported only).

- **Output Quality & Trustworthiness:**
  - **RULER Benchmark (128K context, Table 1):**
    - *Llama-3-8B-1M:* Full Attention = 86.68% average $\rightarrow$ ShadowKV = **86.88%** (outperforms baselines: Quest = 82.03%, InfiniGen = 70.13%, Loki = 9.33%).
    - *GLM-4-9B-1M:* Full Attention = 86.82% average $\rightarrow$ ShadowKV = **85.62%** (Quest = 77.86%, InfiniGen = 67.60%, Loki = 28.57%).
    - *Llama-3.1-8B:* Full Attention = 85.53% average $\rightarrow$ ShadowKV = **83.57%** (Quest = 81.47%, InfiniGen = 72.91%).
    - *Yi-9B-200K:* Full Attention = 65.73% average $\rightarrow$ ShadowKV = **65.53%** (Quest = 57.89%, InfiniGen = 49.31%).
  - **LongBench (>4K context tasks, Table 1):**
    - *Llama-3-8B-1M:* Full Attention = 39.86% average $\rightarrow$ ShadowKV = **39.94%** (Quest = 36.65%, InfiniGen = 31.81%, Loki = 15.78%).
    - *GLM-4-9B-1M:* Full Attention = 48.24% average $\rightarrow$ ShadowKV = **47.89%** (Quest = 44.53%, InfiniGen = 46.31%).
    - *Llama-3.1-8B:* Full Attention = 48.96% average $\rightarrow$ ShadowKV = **48.13%** (Quest = 46.32%, InfiniGen = 48.44%).
  - **Needle In A Haystack (NIAH up to 1M context, Figures 6, 10, 11):**
    - Achieves **100% green retrieval** across all depth percentages (0% to 100%) and context lengths up to 1M on Llama-3-8B-1M and Llama-3-70B-1M.
  - **Multi-Turn Needle Retrieval (Figure 7):**
    - ShadowKV maintains **100% retrieval accuracy across all 8 conversation turns**, whereas eviction baselines (SnapKV) collapse to 0% retrieval starting immediately at turn 2 because evicted tokens cannot be recovered.
  - **FP8 Numerical Robustness (torch.float8_e5m2, Tables 9 & 10):**
    - On RULER (128K): ShadowKV achieves **85.95%** (matching full FP8 baseline of 84.94%).
    - On LongBench: ShadowKV achieves **39.25%** (matching full FP8 baseline of 39.49%).

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Fully compliant. Model weights remain 100% untouched in native BF16 precision. Low-rank compression is applied strictly to pre-RoPE key activations during inference, preserving full weight dynamics.
  - **Dense GQA Compatibility:** Fully compatible. Evaluated directly on Grouped-Query Attention architectures (8:1 and 4:1 GQA models including Llama-3.1-8B, GLM-4-9B, and Yi-9B). In Algorithm 2, landmark scores are summed across sequence lengths and max-pooled across query heads assigned to each KV head ($S_2 = \max_{\text{kv\_group}}(S_1)$), matching the exact GQA group structure of our pinned models (`Qwen 3 4B`, `Phi-4 mini`, `Ministral 3 3B`, `SmolLM3 3B`, `Qwen 3 1.7B`).

- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - **The PCIe Bandwidth Reality Check:**
    - The paper calculates an equivalent bandwidth of $\mathcal{B}_e = 7.2\text{ TB/s}$ assuming an NVIDIA A100 SXM4 ($B_{\text{GPU}} = 2,000\text{ GB/s}$, $B_{\text{PCIe}} = 31.5\text{ GB/s}$).
    - On our local testbed (RTX 4060 8 GB, PCIe Gen 4 x8):
      $$B_{\text{GPU}} \approx 272\text{ GB/s}, \quad B_{\text{PCIe}} \approx 15.75\text{ GB/s}, \quad \frac{B_{\text{GPU}}}{B_{\text{PCIe}}} \approx \frac{272}{15.75} \approx 17.27$$
      For a 32K context ($S=32,768$), $C=8$, $K=64$ (1.56% budget), $O=12$, and hit rate $\alpha=0.60$:
      $$\text{Denominator} = \frac{32768}{8} + 2(64+12)8 + (0.40 \times 64 \times 8 \times 17.27) = 4096 + 1216 + 3537 = 8,849$$
      $$\mathcal{B}_e = \frac{2 \times 32768 \times 272\text{ GB/s}}{8849} \approx 2,014\text{ GB/s} \approx 2.01\text{ TB/s}$$
      While $2.01\text{ TB/s}$ represents an effective $7.4\times$ speedup over the raw $272\text{ GB/s}$ VRAM bus, the PCIe transfer term ($3,537$) accounts for **40.0% of the entire denominator**! If PCIe throughput degrades due to CPU host memory contention, PCIe transfer stalls will dominate decoding.
  - **VRAM & System RAM Budgeting for 10–15 Concurrent Users:**
    - For `Qwen 3 4B` ($8.06\text{ GB}$ weights in BF16) on our 16 GB RTX 4090 cap:
      - Headroom for KV cache: $16.0 - 8.06 = 7.94\text{ GB}$.
      - Under Full Attention, 10 users at 4K context require $5.63\text{ GB}$ (fitting narrowly), but at 32K context require $45.0\text{ GB}$ (immediate OOM).
      - Under ShadowKV ($7.08\times$ reduction), the GPU-resident KV cache for 10 users at 32K context drops from $45.0\text{ GB}$ to $\approx 6.35\text{ GB}$, comfortably fitting within our $7.94\text{ GB}$ headroom.
      - *Host RAM Demand:* Offloading 10 users' 32K values to host RAM requires $\approx 22.5\text{ GB}$, which exceeds a 16 GB system RAM configuration. For consumer deployment, either concurrent sessions must be bounded to $5\text{--}6$ users, or host RAM must be expanded to 32 GB.

- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** **High**. Requires:
    1. Custom online SVD integration during prefill (either torch.linalg.svd or randomized SVD).
    2. Dedicated CUDA kernels or multi-stream PyTorch pipelines for parallel PCIe value gathering and low-rank key matrix multiplication.
    3. Managing non-contiguous sparse chunk tables.
  - **WP-3 Micro-Benchmarking:**
    - Benchmark the exact latency crossover where prefill SVD computation overhead exceeds standard FlashAttention prefill time across context lengths ($4\text{K} \rightarrow 32\text{K}$).
    - Measure empirical PCIe value transfer latency versus Tensor Core key reconstruction latency across PCIe Gen 4 x8 (RTX 4060) and PCIe Gen 4 x16 (RTX 4090).
  - **WP-6 Engine Integration (`vLLM` PagedAttention / `LMCache`):**
    - *The PagedAttention Block Size Tension:* ShadowKV uses a chunk size of $C = 8$ tokens, while vLLM standardizes on block sizes of $16$ or $32$ tokens. Aligning chunk size to $16$ reduces accuracy slightly (as shown in Figure 9, accuracy drops past chunk size 8), whereas using chunk size 8 within 16-token pages creates intra-block fragmentation.
    - *Adaptation for LMCache:* LMCache already supports decoupled host-to-device streaming via `LMCacheMPConnector`. We can adapt ShadowKV's key-vs-value asymmetry into LMCache by caching pre-RoPE low-rank keys on GPU while delegating value paging to LMCache's host memory backend.
- **Classification for Project:** **Component Inspiration for WP-4 & Core Comparison Baseline** (Provides rigorous empirical proof of Key-vs-Value rank asymmetry, which directly motivates our asymmetric precision and tiered storage policies).
