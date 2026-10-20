# CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  When serving large language models with long documents or extended chat histories, running the initial prompt reading step from scratch takes several seconds because the model has to process thousands of tokens before outputting a single word. While systems can save this work by caching and reusing past Key-Value states, storing all these massive numerical tensors in GPU memory is impossible when serving many users or handling long documents. If the system offloads these caches to separate storage machines or host memory over typical network links, transmitting those giant floating-point tensors takes several gigabytes per request—creating a severe communication bottleneck that can take just as long as, or even longer than, recomputing the prompt from raw text.

- **How CacheGen Solves It (Step-by-Step in Plain Words):**
  1. **Find What Stays Similar Across Neighbors (Anchor and Deltas):** CacheGen groups conversation tokens into small clusters of ten tokens. For each cluster, it keeps the first token at high numerical precision as an anchor point. Instead of saving raw numbers for the remaining nine tokens, it only records the tiny difference between each token and that anchor point. Because adjacent tokens in language share similar internal patterns, these differences hover very close to zero and take far less space to describe.
  2. **Protect Sensitive Early Layers (Tiered Precision):** Language models do not treat all layers equally. Early layers extract foundational grammar and token relationships, while deeper layers capture abstract semantic structure. CacheGen splits the model into three equal layer tiers: it applies fine, high-precision binning to early layers and coarser, lossier binning to deep layers. This slashes data size while preventing early errors from snowballing down the network.
  3. **Compress Repetitive Channels (Custom Bitstream Packing):** Instead of saving tensors as standard arrays of numbers, CacheGen profiles the statistical distributions of features across model layers and channels offline. It feeds the quantized differences into a GPU-accelerated statistical coder (similar to how modern video codecs compress movie frames), squeezing the tensor into a tight bitstream without losing any more information.
  4. **Adapt to Bandwidth on the Fly (Dynamic Chunk Streaming):** CacheGen chops long contexts into independent chunks (around 1,500 tokens each). While transmitting chunks sequentially, it monitors actual transfer speeds. If network bandwidth suddenly drops, it dynamically switches later chunks to a more aggressive compression level or falls back to sending raw text for on-demand GPU prefilling, guaranteeing that the first generated token meets strict latency deadlines.

- **How It Compares to Our Project:**
  CacheGen attacks the **Structural** axis (compression via delta encoding, layer-wise quantization, and arithmetic bitstream encoding) and the **Temporal / Spatial** axis (streaming chunks over bandwidth-constrained links to remote storage). While CacheGen was designed to stream KV tensors between cloud storage servers and GPU workers over standard network links, our project (`kv_co-op_hub`) tackles the **PCIe bus bottleneck between local GPU VRAM and Host CPU RAM** on consumer GPUs (8–16 GB VRAM). CacheGen's core compression insights—anchor-relative delta encoding, layer-varying precision, and channel-grouped bitstream packing—can be directly repurposed to compress evicted cache pages before sending them across the PCIe bus, multiplying our effective CPU offload bandwidth.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Reduces KV cache transmission footprint by 3.5x to 4.3x compared to standard quantization; cuts Time-To-First-Token by 3.1x to 4.7x compared to text recomputation, and by 3.2x to 3.7x compared to standard quantization baselines; maintains downstream task quality with under 2% accuracy loss and less than 0.1 perplexity change.
  - **What You Pay:** Requires custom GPU encoding/decoding kernels (adding roughly 200 milliseconds during offline ingestion and modest GPU compute during ingestion); context must be split into chunks of at least 1,000–1,500 tokens to amortize batching and bitstream overheads.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Anchor-Based Delta Encoding Over Chained Deltas:** Consecutive tokens have correlated Key-Value states (variance drops by 2.4x to 2.9x when measuring deltas). Instead of calculating progressive chained differences (token 2 minus token 1, token 3 minus token 2), CacheGen references every token in a 10-token window back to the *same single anchor token*. This architectural choice is brilliant for GPU implementation: it eliminates serial dependency loops, allowing all deltas in a chunk to be compressed and decompressed in parallel across CUDA warps.
  2. **Layer-Varying Quantization Budgets:** Early transformer layers are hypersensitive to quantization noise because errors introduced at layer 1 compound through subsequent attention operations. By partitioning layers into three groups and allocating higher bitwidths / tighter bins to the bottom third, CacheGen achieves dramatic overall size reduction while preserving output fidelity. We can mirror this directly in WP-4 by assigning conservative precision (8-bit or unquantized BF16) to shallow layers and aggressive precision (3-bit or 4-bit) to deep layers.
  3. **Channel-Layer Distribution Grouping for Entropy Coding:** Grouping tensor elements by layer and channel yields significantly lower information entropy than grouping by token position. Profiling separate probability distributions per layer and channel offline allows lossless statistical coders to achieve up to 53% smaller bitstreams than applying a single global distribution.
  4. **Orthogonality to Token Pruning:** CacheGen was evaluated directly on top of token-eviction algorithms like H2O and LLMLingua. It compressed the remaining surviving tokens by an additional 3.3x to 4.2x without degrading accuracy. This confirms that structural bitstream compression and structural token pruning are complementary axes that can be stacked together.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Evaluated on High-End Server Hardware (NVIDIA A40):** CacheGen was benchmarked on an enterprise node with four NVIDIA A40 GPUs (48 GB VRAM each) and dual Xeon Gold processors. On consumer setups (RTX 4060 with 8 GB VRAM or RTX 4090 capped at 16 GB), running extra GPU dequantization and arithmetic decoding kernels consumes precious VRAM and execution units that are already constrained during decode.
  - **Focus on Network Links Rather than Host RAM / PCIe:** CacheGen assumes the cache source is a remote storage server over 1–10 Gbps Ethernet. In single-node consumer serving, the cache lives in Host CPU RAM over PCIe Gen4 (achieving 8–16 GB/s, or 64–128 Gbps). At PCIe speeds, the balance between GPU decompression kernel runtime and raw data transfer time shifts significantly: if a CPU or GPU decompression kernel takes too long, it may be faster to stream uncompressed INT8 or FP8 tensors directly over PCIe rather than running complex arithmetic bitstream decoding.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 7 in `manual_reviews.md` correctly identified all three empirical observations: token-wise delta correlation along the channel axis, layer-varying loss sensitivity where early layers require lossless/higher precision, and channel-layer chunking yielding lower entropy for codec-style bitstream compression.
  - *Refinement from PDF:* The team note mentioned "token that lives near others tends to have similar value... so we can use only first value and the difference to the other". The PDF specifically clarifies that CacheGen uses a fixed anchor token per 10-token group rather than chained token-to-token deltas, specifically to allow massive GPU thread-level parallelism during decoding. Furthermore, CacheGen defaults to 1.5K-token streaming chunks and pipelined CUDA arithmetic decoding to overlap transmission with decompression.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, Junchen Jiang (University of Chicago, Microsoft, Stanford University).
- **Publication / Venue & Year:** ACM SIGCOMM 2024, August 4–8, 2024, Sydney, NSW, Australia (ACM ISBN 979-8-4007-0614-1/24/08, DOI: 10.1145/3651890.3672274; arXiv:2310.07240v6, July 2024).
- **Primary Axis:** Structural (Delta residual calculation, layer-varying quantization, and arithmetic bitstream entropy encoding).
- **Secondary Interacting Axis:** Temporal / Spatial (Dynamic network streaming adaptation, chunk-level latency deadline enforcement, and tiered disaggregated cache storage).
- **Official Codebase / Repository:** [https://github.com/UChi-JCL/CacheGen](https://github.com/UChi-JCL/CacheGen)
- **Baseline Implementation Stack:** Python 2K LoC, CUDA 1K LoC, PyTorch v2.0, CUDA 12.0, HuggingFace Transformers, vLLM (xFormers backend), LangChain integration; evaluated on Mistral-7B, Llama-13B, Llama-33B/34B, and Llama-70B across LongChat, TriviaQA, NarrativeQA, and WikiText.

---

## 3. How the Algorithm Actually Works (Under the Hood)

- **Data Structures Added / Modified:**
  - **Anchor Token Tensor ($K_0, V_0$):** High-precision (8-bit vector-wise quantized) Key and Value tensors for the initial token of each token group (default group size $G = 10$).
  - **Delta Residual Tensors ($\Delta K, \Delta V$):** Floating-point difference tensors between non-anchor tokens and their group anchor.
  - **Quantized Symbol Tensors ($S_K, S_V$):** Discrete integer indices generated after applying layer-specific quantization bin sizes to delta tensors.
  - **Channel-Layer Probability Tables ($P_{\ell, c}$):** Offline-profiled empirical frequency distributions for each layer $\ell$ and channel $c$, consumed by the arithmetic coder.
  - **Context Chunk Bitstream Store:** Partitioned bitstreams stored at multiple quality levels per chunk (default chunk length $C = 1,500$ tokens) mapped by `chunk_id`.

- **Scoring, Compression & Decision Logic:**

  1. *Change-Based Delta Formulation:*
     For a token group $\{t_0, t_1, \dots, t_{G-1}\}$ at layer $\ell \in \{1, \dots, L\}$ and channel $c \in \{1, \dots, D\}$:
     $$\Delta K_{i, \ell, c} = K_{i, \ell, c} - K_{0, \ell, c}, \quad \Delta V_{i, \ell, c} = V_{i, \ell, c} - V_{0, \ell, c} \quad (\forall i \in \{1, \dots, G-1\})$$
     *Plain English:* For every non-anchor token $i$ in the group, subtract the Key/Value vector of the first token $t_0$ from token $i$'s Key/Value vector. Because nearby tokens attend to similar contexts, their representations are close in vector space, driving the variance of $\Delta K$ down by $2.4\times\text{--}2.9\times$ compared to the raw tensor. Referencing token $0$ rather than token $i-1$ allows all 9 deltas to be calculated in parallel without waiting for sequential steps.

  2. *Layer-Wise Quantization & Bin Allocation:*
     The $L$ transformer layers are partitioned into three equal groups: Shallow ($[1, L/3]$), Middle ($[L/3 + 1, 2L/3]$), and Deep ($[2L/3 + 1, L]$). Given layer group index $g \in \{1, 2, 3\}$, the quantized symbol $s$ for delta value $\delta$ is obtained via:
     $$s = \text{round}\left(\frac{\delta}{w_g}\right), \quad \text{where } w_g \in \{w_{\text{shallow}}, w_{\text{mid}}, w_{\text{deep}}\}$$
     *(In CacheGen's default configuration: $w_{\text{shallow}} = 0.5$, $w_{\text{mid}} = 1.0$, $w_{\text{deep}} = 1.5$.)*  
     *Plain English:* Divide the delta value by the layer group's bin width $w_g$ and round to the nearest integer symbol. Early layers receive narrow bins ($w_1 = 0.5$, preserving high fidelity), whereas late layers receive wide bins ($w_3 = 1.5$, tolerating larger rounding error). The anchor token $t_0$ bypasses this formula and is stored at 8-bit precision.

  3. *Arithmetic Coding (AC) Bitstream Compression:*
     Given discrete symbol $s \in \mathcal{S}$ occurring at layer $\ell$ and channel $c$ with profiled probability $P_{\ell, c}(s)$, the theoretical optimal code length in the bitstream is:
     $$\mathcal{L}(s) = -\log_2 P_{\ell, c}(s)$$
     *Plain English:* Frequent delta symbols (which cluster tightly around zero) are assigned fractional bits close to zero, while rare outliers receive longer bit representations. Profiling $P_{\ell, c}$ per channel and layer rather than globally cuts the resulting bitstream size by up to $53\%$.

  4. *Adaptive Chunk Streaming Logic (Algorithm 1):*
     When streaming context chunks $\{1, \dots, M\}$ toward a target Time-To-First-Token deadline ($\text{SLO}$), CacheGen measures effective bandwidth $R$ over the preceding chunk and computes remaining time $t_{\text{rem}} = \text{SLO} - t_{\text{elapsed}}$. For each candidate encoding level $k$ with bitstream size $\text{Size}(\text{chunks\_remaining}, k)$:
     $$\text{Pick } k^* = \arg\max_{k} \left\{ \text{Quality}(k) \;\middle|\; \frac{\text{Size}(\text{chunks\_remaining}, k)}{R} \le t_{\text{rem}} \right\}$$
     If $t_{\text{recompute}} \le t_{\text{rem}}$, CacheGen sends raw text instead to allow zero-loss GPU recomputation.  
     *Plain English:* Choose the highest-precision compression setting whose total transmission time will finish before the deadline based on current transfer speed. If the network is fast enough to transfer text and run full GPU prefill before the deadline, send text to achieve 100% lossless output.

- **Execution Cadence:**
  - Context chunking, delta computation, layer quantization, and arithmetic bitstream encoding are performed **offline during context ingestion** (`store_kv`).
  - Empirical symbol distribution profiling ($P_{\ell, c}$) is executed **once offline per model architecture**.
  - Dynamic rate adaptation and pipelined arithmetic decoding run **incrementally per 1.5K context chunk** when a query requests a stored context.
  - Normal auto-regressive token generation runs completely unmodified on the reconstructed KV tensors.

- **Memory Hierarchy & Placement:**
  - **Keys ($K$) & Values ($V$):** In storage, held as compressed arithmetic bitstreams on secondary storage servers (or Host CPU RAM). Upon arrival at the GPU, custom CUDA kernels dequantize the bitstreams back into contiguous FP16/BF16 tensors ready for attention execution.
  - **Anchor Tokens:** Stored at INT8 precision; non-anchor tokens stored as quantized delta symbols.
  - **Granularity:** Token-group level ($G=10$ tokens) for delta encoding; layer-group level (3 tiers) for quantization binning; chunk level ($C=1,500$ tokens) for network streaming adaptation and GPU decompression pipelining.

- **Decode-Phase Overheads:**
  - GPU arithmetic decoding kernel takes minimal compute compared to prefilling: decoding overhead is pipelined with incoming chunk network transfer, hiding latency behind I/O.
  - Once decoded into GPU VRAM, standard generation runs at native speed with zero dequantization overhead during the auto-regressive decode phase.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

- **Memory Saved (Efficiency):**
  - **Bitstream Compression Ratio:** Achieves **$3.5\times\text{--}4.3\times$ reduction in KV cache transfer size** compared to uniform quantization baselines across tested models and datasets.
  - **Stacking on Token Eviction (H2O & LLMLingua):** When applied to KV caches already pruned by H2O or LLMLingua, CacheGen further compresses the surviving cache by **$3.3\times\text{--}4.2\times$**.
  - **Absolute Cache Footprint Example:** An $8.5\text{K}$-token context for Llama-13B occupies $\approx 5\text{ GB}$ across all compressed multi-rate representations on disk (costing $\$0.05/\text{month}$ on S3 vs. $\$0.00085$ per recomputed query).

- **Speed & Latency (TTFT, TPOT, Tail Latency):**
  - **TTFT Reduction (3 Gbps network link):**
    - Slashes TTFT by **$3.1\times\text{--}4.7\times$** compared to recomputing KV cache from raw text.
    - Slashes TTFT by **$3.2\times\text{--}3.7\times$** compared to uniform quantization baselines.
    - Slashes TTFT by **$1.67\times\text{--}1.81\times$** even when compared against an 8-bit quantization baseline.
  - **SLO Violation Rates (Under fluctuating 0.1–10 Gbps network traces):**
    - At $\text{SLO} = 1.0\text{ s}$: Drops violation rate from **$81\%$** (quantization baseline) down to **$8\%$** (CacheGen).
    - At $\text{SLO} = 0.5\text{ s}$: Achieves a **$60\%$ lower violation rate** than uniform quantization.
  - **Decompression / Offline Delays:**
    - Offline context encoding delay: $\approx 200\text{ ms}$ per context.
    - GPU decoding overhead: Negligible wall-clock penalty due to chunk transmission overlapping (`chunk } i` network receive pipelined with `chunk } i-1` GPU decompression).
  - **Tail Latency (P95/P99):** *Not reported in paper as a continuous distribution curve*; the paper evaluates time-to-first-token mean speedups, SLO violation percentages, and step-by-step time-series traces under dynamic bandwidth.

- **Output Quality & Trustworthiness:**
  - **LongChat ($9.4\text{K}$ tokens, Topic Retrieval Accuracy):**
    - Llama-70B: Maintains $>95\%$ accuracy; quality degradation from lossy compression is **$\le 2\%$**.
    - Mistral-7B & Llama-33B: Near-identical accuracy curves between full KV and CacheGen across TTFT operating points.
  - **TriviaQA & NarrativeQA ($9.3\text{K}\text{--}14\text{K}$ tokens, Reading Comprehension F1):**
    - F1 score degradation is **$< 0.1\%$** compared to lossless full-precision KV cache.
  - **WikiText ($5.9\text{K}$ tokens, Language Modeling Perplexity):**
    - Perplexity increase ($\Delta\text{PPL}$) is **$< 0.1$ points** across all evaluations.

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Fully compliant. Model parameters remain pristine in native BF16; compression operates solely on the intermediate Key-Value activation tensors.
  - **Dense GQA Compatibility:** Fully compatible with our pinned models (**Qwen 3 4B**, **Phi-4 mini**, **Ministral 3 3B**, **SmolLM3 3B**, **Qwen 3 1.7B**). Grouped-Query Attention simply reduces the number of Key/Value heads ($H_{KV}$), which means each token has fewer channels to compress. The delta formulation and layer grouping operate identically on GQA head dimensions ($D = 128$).

- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - **The Bandwidth Domain Shift:** CacheGen targets 1–10 Gbps cloud networks ($0.125\text{--}1.25\text{ GB/s}$). On consumer workstations, the transfer bottleneck is PCIe Gen4 x8 or x16 ($8\text{--}32\text{ GB/s}$). Because PCIe is $10\times\text{--}50\times$ faster than Gigabit Ethernet, the compute cost of running complex arithmetic decoding on GPU CUDA cores can become a bottleneck relative to raw PCIe throughput.
  - **Memory Footprint Feasibility:** For Qwen 3 4B ($144\text{ KiB/token}$ in native BF16), a 32K context requires $4.50\text{ GB}$ of uncompressed VRAM. Compressing inactive cache pages with CacheGen's $3.5\times$ factor drops this footprint to $\approx 1.28\text{ GB}$, allowing our 16 GB RTX 4090 cap to comfortably host multi-session SME traffic (10 concurrent users @ 4K context would take only $\approx 1.6\text{ GB}$ compressed vs. $5.63\text{ GB}$ uncompressed).

- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** High for full end-to-end implementation (requires custom CUDA entropy coders); Medium for structural quantization and delta encoding alone.
  - **WP-3 Micro-Benchmarking:** We can implement a clean micro-benchmark evaluating:
    1. The variance reduction of anchor-based delta encoding across our pinned GQA models.
    2. The empirical accuracy impact of CacheGen's 3-tier layer quantization ($0.5, 1.0, 1.5$ binning) on WikiText-2 and LongMemEval.
    3. Host-to-device PCIe transfer latency comparing raw BF16 vs. INT8/INT4 vector-quantized vs. entropy-coded bitstreams.
  - **WP-6 Engine Integration (`vLLM` PagedAttention / `LMCache`):**
    - *Paged Memory Synergy:* vLLM stores KV caches in non-contiguous memory blocks (e.g., 16 tokens per block). CacheGen's 10-token grouping aligns naturally with block-based storage! We can set the anchor token to the first token of each PagedAttention block and compress blocks individually.
    - *LMCache Connector Integration:* Rather than hacking vLLM's internal attention kernels, CacheGen is best integrated directly inside `LMCache` as a custom serialization format for remote/host storage connectors (`LMCacheMPConnector`), decompressing blocks asynchronously into vLLM page tables as they arrive over PCIe.

- **Classification for Project:** **Component Inspiration for WP-4 & Comparison Baseline for WP-3** (Provides foundational mechanisms for layer-varying bit allocation, anchor-relative delta calculations, and tiered offloading bandwidth management).
