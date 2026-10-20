# QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  Everyone assumed that quantizing language models down to 4-bit integers would automatically double inference speeds because modern graphics cards have dedicated hardware designed to crunch 4-bit numbers twice as fast as 8-bit numbers. In reality, existing 4-bit systems like Atom and QuaRot turned out to be 20% to 25% *slower* than standard 8-bit systems during high-volume serving. The hidden culprit is that graphics chips have two different processing components: lightning-fast matrix engines (Tensor Cores) and slower, general-purpose calculation units (CUDA Cores). When numbers are stored in 4 bits, the chip has to unpack and convert them back into larger numbers before doing the math. This unpacking process ("dequantization") runs entirely on the slow general-purpose units, creating a massive traffic jam that eats up 20% to 90% of total execution time. On top of that, squishing the Key-Value (KV) cache down to 4 bits normally causes severe accuracy loss because certain channels develop massive activation spikes ("outliers").

- **How It Solves It (Step-by-Step in Plain Words):**
  1. **Before / During Prefill (Smoothing Out Outliers):**
     The authors discovered an important pattern: Value vectors in the cache are naturally smooth and well-behaved, but Key vectors consistently have sharp outlier spikes in specific channels that are roughly ten times larger than normal values. To fix this without hurting performance, they introduce *SmoothAttention*. They scale down these spiky outlier channels in the Keys using a calculated channel divisor, and simultaneously scale up the Query vectors by the exact same amount. When the Queries and Keys are multiplied during attention, the multipliers perfectly cancel out. Crucially, they pair up channels to match how rotary position embeddings rotate data, allowing them to bake this smoothing directly into the model's existing projection weights offline before serving starts, adding zero extra computation steps during inference.
  2. **Where the Data Lives (Storage & Compression):**
     Model weights are stored in 4-bit integers on GPU video memory. Intermediate activations between layers are converted to 8-bit integers. The KV cache is stored in 4-bit integers within paged memory blocks on the GPU. Small dynamic 16-bit floating-point scale factors and zero offsets are stored right beside each attention head's tokens inside the memory pages, ensuring the cache stays tiny without losing dynamic range.
  3. **During Decode (Generating New Tokens):**
     When generating tokens, QServe uses custom-engineered GPU operations to bypass the slow general-purpose units:
     - *Progressive Weight Unpacking:* It stores weights in a rearranged layout so a single instruction can unpack four 4-bit weights simultaneously. By restructuring the math to perform zero-point subtraction *after* multiplication rather than before, intermediate numbers never exceed 8 bits, completely preventing math overflows and eliminating extra conversion instructions.
     - *Accelerated 4-Bit Attention:* Instead of running five separate mathematical operations per token to unpack 4-bit cached keys and values, it uses clever bit-shifting tricks to do it in just two operations. It switches intermediate attention math to 16-bit floating-point instead of 32-bit (doubling the calculation ceiling) and prefetches scales and zero points in the background while the previous step finishes.

- **How It Compares to Our Project:**
  Like our project's **Structural** axis, QServe aims to compress the KV cache to 4 bits while eliminating GPU processing bottlenecks. However, QServe quantizes the entire system (weights to 4-bit, activations to 8-bit, and all KV tokens uniformly to 4-bit). In contrast, our project mandates strictly unmodified 16-bit model weights, keeps high-priority conversational tokens at full 16-bit quality, and applies 4-bit quantization selectively to lower-priority tokens before offloading them to system memory. QServe's SmoothAttention mechanism and its low-overhead 4-bit attention kernel provide the exact mathematical and systems blueprint we need for our selective quantization tier.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Slashes KV cache memory usage by 50% compared to 8-bit and by 75% compared to 16-bit, doubling achievable serving batch sizes under the same GPU memory budget; delivers 1.2x to 3.5x higher token generation throughput than NVIDIA TensorRT-LLM on A100 and L40S GPUs; preserves accuracy across long-context benchmarks (less than 0.2 points drop on LongBench); allows an 8,000-dollar workstation GPU (L40S) to outperform a 25,000-dollar datacenter GPU (A100), cutting hardware costs by 3x.
  - **What You Pay:** Requires custom GPU kernels written in CUDA and assembly to realize speedups; requires a one-time offline calibration pass on a small text sample to compute key smoothing factors and reorder weights; does not perform CPU offloading or selective priority tiering.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Key Outliers vs. Value Smoothness:** Values do not suffer from activation outliers, whereas Keys exhibit persistent, fixed outlier channels (roughly ten times larger in magnitude) per head across all layers. This means Value vectors can be aggressively quantized with simple uniform scaling, but Key vectors require outlier smoothing before 4-bit compression to prevent attention scores from collapsing.
  2. **RoPE-Commutative Channel Pairing:** Modern open models apply Rotary Position Embeddings (RoPE) to Keys and Queries by rotating pairs of adjacent channels (specifically channel $i$ and channel $i + D/2$ within each head). If you scale channel $i$ independently, RoPE breaks. QServe solves this by setting the scale factor of channel $i$ equal to the maximum of channel $i$ and channel $i + D/2$. This makes the smoothing operation commutative with RoPE, meaning we can fold the scales directly into the Key and Query projection matrices offline with zero runtime latency penalty!
  3. **The CUDA Core Dequantization Trap:** Naively unpacking 4-bit KV caches requires five arithmetic operations per element (isolate bits via mask, bit-shift, integer-to-float conversion, floating-point multiply, and subtract). On modern GPUs, this dequantization alone saturates the CUDA core instruction ceiling, making naive 4-bit attention *slower* than 8-bit attention. To achieve true wall-clock speedups, we must adopt bit tricks that unpack elements in two operations, switch attention arithmetic to 16-bit floating point, and prefetch scales asynchronously.
  4. **Dynamic Per-Head Paged Scaling:** Rather than using coarse tensor-wide static scales, QServe packs dynamic per-head scale factors and zero offsets directly into each paged memory block immediately adjacent to the token tensors. This maintains spatial memory locality and fits cleanly into paged virtual memory architectures.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Evaluated on High-End Enterprise GPUs (A100 SXM4 & L40S):** Both GPUs evaluated have massive memory buses (A100 has 2,039 GB/s HBM2e) or massive tensor compute (L40S has 733 TFLOPS FP16 and 91.6 TFLOPS FP32). While the paper shows Ada Lovelace GPUs (L40S) handle dequantization much better than Ampere (A100) due to higher CUDA core throughput, our consumer GPUs (RTX 4060 with 8 GB VRAM and RTX 4090 capped at 16 GB VRAM) have narrower memory buses (128-bit on RTX 4060, 272 GB/s) and limited PCIe bandwidth (Gen 4 x8, ~16 GB/s), making kernel fusion and register allocation even more constrained.
  - **Uniform Quantization Across All Tokens:** QServe squishes every token in the KV cache to 4 bits. In multi-turn dialogue, needle-in-a-haystack tasks, or sensitive reasoning steps, uniform 4-bit quantization can degrade fine-grained token distinction. A selective tiering policy that leaves attention sinks and high-priority tokens in full 16-bit precision provides a safer quality guarantee.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 28 of our manual notes correctly identified the core contribution: W4A8KV4 quantization with progressive quantization and SmoothAttention, and pinpointed that CUDA core dequantization overhead (20–90%) wipes out theoretical gains unless fused directly into register-level operations.
  - *Clarification from PDF:* The manual review notes listed datasets as "Llama-2/3, Qwen1.5-72B, LongBench, WikiText-2." The full PDF also benchmarks Llama-1 (7B, 13B, 30B), Mistral-7B, Mixtral-8x7B, and Yi-34B across both A100 and L40S architectures. In addition, the note suggested weight quantization is mandatory; we clarify that SmoothAttention and the optimized KV4 attention kernel are modular and directly applicable to our selective KV-cache tier without quantizing model weights.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Yujun Lin*, Haotian Tang*, Shang Yang*, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han (MIT, UMass Amherst, MIT-IBM Watson AI Lab, NVIDIA) (*Equal contribution)
- **Publication / Venue & Year:** Proceedings of the 8th MLSys Conference (MLSys 2025), Santa Clara, CA, USA, 2025. (arXiv:2405.04532v3, May 2025)
- **Primary Axis:** Structural (Progressive weight-activation quantization, SmoothAttention outlier mitigation, and dynamic per-head 4-bit KV cache quantization)
- **Secondary Interacting Axis:** Engine & Kernel Fusion (Compute-aware weight reordering, register-level parallelism via `vadd4`, and fused low-overhead KV4 attention kernels)
- **Official Codebase / Repository:** [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve) (and [hanlab.mit.edu/projects/qserve](https://hanlab.mit.edu/projects/qserve))
- **Baseline Implementation Stack:** Custom CUDA and PTX assembly kernels integrated with a PyTorch 2.2.0 / Hugging Face frontend. Evaluated against TensorRT-LLM v0.9.0 (FP16, W8A8, W4A16), Atom (W4A4), and QuaRot (W4A4).

---

## 3. How the Algorithm Actually Works (Under the Hood)

```
[LayerNorm / Activation]
         │ (Fuses INT8 Activation Quantization)
         ▼
[W4A8 GEMM Projection] ──► Uses Compute-Aware Reordering (128-bit loads)
         │              ──► Subtraction after Multiplication via vadd4 (RLP)
         │              ──► Epilogue scales & zero subtraction: t_X = X * 1_k
         ▼
[SmoothAttention Scaling] ──► Fused offline into W_Q and W_K (RoPE-commutative)
         │
         ▼
[Dynamic Per-Head KV4] ──► Keys & Values compressed to INT4
         │              ──► FP16 scale and zero packed into Paged KV blocks
         ▼
[Accelerated KV4 Attention] ──► Unpacking in 2 ALU ops (Kim et al. bit-tricks)
         │                  ──► FP16 accumulation, async prefetching of scales
         ▼
[FP16 Block Output]
```

### Data Structures Added / Modified

1. **Dynamic Paged KV4 Block Table:** Extends standard PagedAttention virtual block tables. Each physical block stores Keys and Values in 4-bit unsigned integers (`UINT4`). Immediately adjacent to the packed feature payloads, each block stores dynamic per-head scaling factors ($s \in \text{FP16}$) and zero points ($z \in \text{FP16}$), updated on-the-fly during decoding.
2. **Compute-Aware Packed Weight Layout:** Model weights are offline-reordered into $32 \times 32$ tiles and packed into 128-bit words per thread. Every 32 `UINT4` weights $(w_0, w_1, \dots, w_{31})$ are interleaved as $(w_0, w_{16}, w_1, w_{17}, \dots)$ to align with GPU register-level SIMD operations without requiring runtime pointer arithmetic.

### Scoring, Compression & Decision Logic

#### 1. SmoothAttention for 4-bit Key Cache
To eliminate outlier channels in Keys without quantizing Queries, SmoothAttention applies a per-channel diagonal scaling matrix $\Lambda = \text{diag}(\lambda)$:

$$Z = (Q \Lambda) \cdot (K \Lambda^{-1})^T$$

*Plain English:* We multiply the Query matrix $Q$ by a diagonal scaling matrix $\Lambda$ and divide the Key matrix $K$ by the exact same scaling matrix. When computing the dot product attention $Q K^T$, the scaling factors cancel out exactly, preserving mathematical equivalence while smoothing out the large spikes in $K$.

Because Queries are not stored in the KV cache, the per-channel scale factor $\lambda_i$ focuses entirely on suppressing Key outliers:

$$\lambda_i = \max(|K_i|)^\alpha \quad (\text{empirically } \alpha = 0.5)$$

*Plain English:* For channel $i$, find the maximum absolute Key activation value across tokens and raise it to power $\alpha = 0.5$ (taking the square root). This scales down extreme peaks while preventing smaller channels from underflowing.

#### 2. RoPE-Commutative Constraint
Modern LLMs apply Rotary Position Embeddings (RoPE), which pair channel $i$ with channel $i + D/2$ within each head of dimension $D$. To allow fusing $\Lambda$ into preceding linear layers without breaking RoPE, the scales of paired channels must be identical:

$$\lambda_i = \lambda_{i + D/2} = \max\left(\max(|K_i|), \max(|K_{i + D/2}|)\right)^\alpha$$

*Plain English:* To make sure the scaling can commute with the rotary embedding rotation, channel $i$ and channel $i + D/2$ share the exact same scale factor, calculated from the maximum value found in either channel.

The scaling matrices are then folded offline into the projection layer weights:

$$W_Q^{\text{fused}} = \Lambda W_Q, \quad W_K^{\text{fused}} = \Lambda^{-1} W_K$$

*Plain English:* Multiply the Query projection weights by $\Lambda$ and divide the Key projection weights by $\Lambda$ ahead of time. At runtime, the model produces pre-smoothed Keys and pre-scaled Queries with zero extra latency or memory traffic.

#### 3. Progressive Group Quantization (Weights)
To achieve INT8 Tensor Core execution for 4-bit weights, weights $W \in \mathbb{R}^{k \times n}$ are quantized progressively:

$$\hat{W} = Q_{Ws8}^{(0)} \cdot s_{\text{fp16}}^{(0)}$$

$$Q_{Ws8}^{(0)} = (Q_{Wu4} - z_{u4}) \cdot s_{u8}^{(1)}$$

*Plain English:* Weights are first converted to an intermediate 8-bit integer representation $Q_{Ws8}^{(0)}$ with per-channel floating-point scales $s_{\text{fp16}}^{(0)}$. That intermediate tensor is then quantized into unsigned 4-bit integers $Q_{Wu4}$ using unsigned 8-bit group scales $s_{u8}^{(1)}$ and 4-bit zero points $z_{u4}$.

To guarantee that dequantized intermediate values strictly fit in the signed 8-bit integer range $[-128, 127]$ without costly hardware saturation instructions, the symmetric INT8 quantization range is constrained to a **protective range**:

$$q_{s8} \le 127 - \frac{1}{2} s_{u8} \implies q_{s8} \in [-119, 119]$$

*Plain English:* By shrinking the initial 8-bit quantization range from $[-127, 127]$ down to $[-119, 119]$, the second-level dequantization can never mathematically exceed $127$, preventing arithmetic overflow during fast integer math without slowing down the GPU.

#### 4. Subtraction After Multiplication & Epilogue Fusion
For per-channel W4A8 GEMM, zero-point subtraction is moved out of the main loop and fused into the epilogue:

$$O = \hat{X}\hat{W} = (Q_X \odot S_X)((Q_W - Z_W) \odot S_W)$$

$$O = (Q_X Q_W) \odot (s_W \times s_X) - t_X \times (z_W \odot s_W), \quad \text{where } t_X = X \mathbf{1}_k$$

*Plain English:* Instead of subtracting zero points inside the high-frequency GEMM loop, we perform the pure matrix multiplication first on Tensor Cores, and subtract the zero-point correction in the final write-out step (epilogue). The input token sum $t_X$ is precomputed inside the preceding memory-bound normalization layer with near-zero latency overhead.

#### 5. Register-Level Parallelism via `vadd4`
In per-group quantization, intermediate subtraction after multiplication enables SIMD vector instructions:
- NVIDIA GPUs provide the `vadd4` instruction, which performs four 8-bit integer additions simultaneously in a single 32-bit register operation.
- Because progressive quantization guarantees that intermediate scaled values do not overflow 8 bits, four weights can be unpacked and adjusted in parallel in registers, cutting dequantization overhead down to match standard W4A16 kernels.

#### 6. Accelerated KV4 Attention Kernel
To eliminate the CUDA core bottleneck in decode attention:
1. **Instruction Reduction via Bit Tricks:** Applies bitwise masking and shifting (Kim et al., 2022) to unpack 4-bit cached values into floating-point registers in only 2 ALU operations per element (down from 5 ALU operations in naive implementations).
2. **FP16 Arithmetic Ceiling:** Replaces all FP32 accumulator and intermediate operations in TensorRT-LLM attention with FP16 equivalents, doubling the theoretical compute roofline from 9.8 Ops/Byte to 19.6 Ops/Byte.
3. **Asynchronous Prefetching:** Prefetches the scaling factors and zero points for the next attention tile while computing the current tile, hiding address calculation latency.

### Execution Cadence
- **Offline / Initialization:** Channel outlier profiling, SmoothAttention weight folding ($W_Q, W_K$), progressive weight quantization, and compute-aware weight reordering run once before serving.
- **Per-Token Decode Step:** Dynamic per-head scale and zero calculation occurs during Key/Value cache writing. Accelerated KV4 attention execution runs once per generated token on CUDA cores.

### Memory Hierarchy & Placement
- **Keys ($K$) & Values ($V$):** Stored entirely on GPU VRAM inside 4-bit unsigned integer (`UINT4`) paged blocks. Per-head scales and zero points are stored in FP16 inside the block headers.
- **Model Weights ($W$):** Stored in GPU VRAM as packed `UINT4` tiles with `UINT8` group scales and `UINT4` zero points.
- **Activations ($X$):** Quantized on-the-fly to signed 8-bit integers (`SINT8`) inside fused LayerNorm / activation kernels before entering GEMMs.
- **Granularity:** Per-group weight quantization ($g=128$) or per-channel; per-token activation quantization; per-head dynamic KV cache quantization.

### Decode-Phase Overheads
- **Host-GPU PCIe Transfers:** None. All quantized weights and KV blocks remain resident in GPU VRAM.
- **CUDA Core Dequantization:** In naive 4-bit attention, dequantization saturates CUDA cores, causing a 1.1x–1.2x slowdown over 8-bit KV on A100. QServe's bit-trick and FP16 optimizations cut fused attention latency from $0.48\text{ ms}$ down to $0.28\text{ ms}$ ($1.71\times$ faster), achieving a net $1.5\times$ speedup over TensorRT-LLM KV8.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

### Memory Saved (Efficiency)
- **KV Cache Memory Footprint:** 4-bit KV cache reduces KV memory by $2\times$ relative to KV8 and $4\times$ relative to native FP16/BF16.
- **Batch Size Scaling:** On an L40S GPU (48 GB VRAM), QServe fits the identical batch sizes and concurrent requests that previously required an 80 GB A100 under TensorRT-LLM.
- **Static Model Weight Savings:** W4 weights occupy $\approx 4.5\text{ GB}$ for 8B models (vs. $16\text{ GB}$ in BF16) and $\approx 38\text{ GB}$ for 72B models (vs. $144\text{ GB}$ in BF16).

### Speed & Latency (Throughput & Micro-Kernel Latency)

#### 1. End-to-End Generation Throughput (Tokens / Second)
*Measured with input sequence length 1,024 and output sequence length 512 under identical memory budgets (48 GB on L40S, 80 GB on A100) (Table 4):*

| Hardware | Serving Framework & Precision | Llama-3 8B | Llama-2 7B | Mistral 7B | Llama-2 13B | LLaMA 30B | Yi 34B | Llama-2 70B | Qwen1.5 72B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L40S (48GB)** | TRT-LLM FP16 | 1,326 | 444 | 1,566 | 92 | OOM | OOM | OOM | OOM |
| | TRT-LLM W4A16 | 1,431 | 681 | 1,457 | 368 | 148 | 313 | 119 | 17 |
| | TRT-LLM W8A8 | 2,634 | 1,271 | 2,569 | 440 | 123 | 364 | OOM | OOM |
| | **QServe W4A8KV4 (Ours)** | **3,656** | **2,394** | **3,774** | **1,327** | **504** | **869** | **286** | **59** |
| | *Measured Speedup vs. Best TRT-LLM* | **$1.39\times$** | **$1.88\times$** | **$1.47\times$** | **$3.02\times$** | **$3.41\times$** | **$2.39\times$** | **$2.40\times$** | **$3.47\times$** |
| **A100 (80GB)** | TRT-LLM FP16 | 2,503 | 1,549 | 2,371 | 488 | 80 | 145 | OOM | OOM |
| | TRT-LLM W4A16 | 2,370 | 1,549 | 2,403 | 871 | 352 | 569 | 358 | 143 |
| | TRT-LLM W8A8 | 2,396 | 2,334 | 2,427 | 1,277 | 361 | 649 | 234 | 53 |
| | **QServe W4A8KV4 (Ours)** | **3,005** | **2,908** | **2,970** | **1,741** | **749** | **797** | **419** | **340** |
| | *Measured Speedup vs. Best TRT-LLM* | **$1.20\times$** | **$1.25\times$** | **$1.22\times$** | **$1.36\times$** | **$2.07\times$** | **$1.23\times$** | **$1.17\times$** | **$2.38\times$** |

> [!IMPORTANT]
> **Cross-Hardware Cost Inversion:** QServe serving Llama-3-8B on an L40S GPU ($8,000 retail) achieves **3,656 tokens/s**, which is **$1.46\times$ faster** than TensorRT-LLM running on an A100 GPU ($25,000 retail) at 2,503 tokens/s, effectively slashing serving hardware capital cost by **$3\times$**.

#### 2. Attention Micro-Kernel Latency on A100 (Table 1)
*Decoding stage attention kernel execution time across sequence lengths:*

| Sequence Length | TRT-LLM 8-bit KV | Naive 4-bit KV | QServe 4-bit KV (Ours) | Relative Speedup vs. TRT-LLM KV8 |
| :--- | :--- | :--- | :--- | :--- |
| **128** | $0.09\text{ ms}$ | $0.10\text{ ms}$ ($0.87\times$) | **$0.07\text{ ms}$** | **$1.29\times$** |
| **256** | $0.14\text{ ms}$ | $0.16\text{ ms}$ ($0.86\times$) | **$0.11\text{ ms}$** | **$1.32\times$** |
| **512** | $0.23\text{ ms}$ | $0.27\text{ ms}$ ($0.87\times$) | **$0.16\text{ ms}$** | **$1.44\times$** |
| **1024** | $0.42\text{ ms}$ | $0.48\text{ ms}$ ($0.88\times$) | **$0.28\text{ ms}$** | **$1.49\times$** |
| **1536** | $0.62\text{ ms}$ | $0.69\text{ ms}$ ($0.90\times$) | **$0.41\text{ ms}$** | **$1.51\times$** |

- **Step-by-Step Optimization Breakdown ($64 \times 1024$ input on A100):**
  - Baseline naive KV4 attention: $0.48\text{ ms}$
  - Applying Kim et al. bit tricks: $0.44\text{ ms}$ ($-0.04\text{ ms}$)
  - Simplifying control logic: $0.39\text{ ms}$ ($-0.05\text{ ms}$)
  - Converting $QK$ and $SV$ products to FP16: $0.33\text{ ms}$ ($-0.06\text{ ms}$)
  - Asynchronous scale/zero prefetching: **$0.28\text{ ms}$** ($-0.05\text{ ms}$, total **$1.71\times$ kernel speedup**)
- **Tail Latency (P95/P99):** Not reported in paper (mean throughput and kernel execution times only).

### Output Quality & Trustworthiness

#### 1. Language Modeling Perplexity on WikiText-2 (Sequence Length 2048, Table 2)

| Precision & Method | Llama-3 8B | Llama-2 7B | Llama-2 13B | Llama-2 70B | Mistral 7B | Mixtral 8x7B | Yi 34B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | **6.14** | **5.47** | **4.88** | **3.32** | **5.25** | **3.84** | **4.60** |
| **W8A8 SmoothQuant** | 6.28 | 5.54 | 4.95 | 3.36 | 5.29 | 3.89 | 4.69 |
| **W4A16 AWQ ($g=128$)** | 6.54 | 5.60 | 4.97 | 3.41 | 5.37 | 4.02 | 4.67 |
| **W4A4 QuaRot ($g=128$)** | 8.20 | 6.10 | 5.40 | 3.79 | 5.71 | NaN | NaN |
| **W4A4 Atom ($g=128$)** | 7.57 | 6.03 | 5.27 | 3.69 | 5.66 | 4.42 | 4.92 |
| **QoQ W4A8KV4 ($g=128$)** | **6.70** | **5.67** | **5.06** | **3.46** | **5.41** | **4.13** | **4.73** |

*Key finding:* QoQ with 4-bit KV cache incurs only a $+0.20$ perplexity delta over FP16 on Llama-2-7B ($5.47 \to 5.67$), substantially outperforming W4A4 baselines QuaRot ($6.10$) and Atom ($6.03$).

#### 2. LongBench Evaluation (Llama-3.1-8B-Instruct, Table 5)
*QoQ W4A8KV4 ($g=128$) vs. Native BF16 across long-context tasks:*
- **QASPER:** BF16 $35.07$ vs. QoQ $35.45$ ($+0.38$)
- **GovReport:** BF16 $34.54$ vs. QoQ $34.09$ ($-0.45$)
- **HotpotQA:** BF16 $16.68$ vs. QoQ $17.46$ ($+0.78$)
- **MultiNews:** BF16 $26.84$ vs. QoQ $26.73$ ($-0.11$)
- **TriviaQA:** BF16 $91.65$ vs. QoQ $91.45$ ($-0.20$)
- **Average LongBench Score:** **BF16 $38.52$ vs. QoQ $38.38$** (Only **$-0.14$ points** overall quality degradation).

#### 3. Common-Sense Zero-Shot Accuracy (Llama-2-7B, Table 3)
- **PIQA:** FP16 $79.05\%$ vs. QoQ $78.07\%$ (QuaRot: $76.77\%$, Atom: $75.14\%$)
- **ARC-Easy:** FP16 $74.58\%$ vs. QoQ $73.32\%$ (QuaRot: $69.87\%$, Atom: $52.99\%$)
- **ARC-Challenge:** FP16 $46.25\%$ vs. QoQ $44.80\%$ (QuaRot: $40.87\%$, Atom: $38.40\%$)
- **HellaSwag:** FP16 $76.05\%$ vs. QoQ $74.98\%$ (QuaRot: $72.16\%$, Atom: $69.37\%$)
- **WinoGrande:** FP16 $69.77\%$ vs. QoQ $68.59\%$ (QuaRot: $63.77\%$, Atom: $62.75\%$)
- **Average:** FP16 $69.14\%$ vs. QoQ $67.95\%$ (QuaRot: $64.69\%$, Atom: $59.73\%$)

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Highly coherent, with a vital architectural distinction. While QServe compresses model weights to W4 to accelerate linear projections in compute-bound batches, our senior project explicitly mandates unmodified BF16 weights to preserve native model reasoning and isolate KV cache behavior. Fortunately, **QServe's KV cache innovations are completely modular**: SmoothAttention operates by scaling Keys and Queries, and the accelerated KV4 attention kernel operates independently of whether weights are INT4 or BF16. We can adopt SmoothAttention and the KV4 attention kernel exclusively for the KV cache while leaving model weights in native BF16 precision.
  - **Dense GQA Compatibility:** Fully compatible. QServe explicitly evaluates and supports Grouped-Query Attention models (Llama-3-8B with 32:8 GQA, Mistral-7B, Qwen1.5-72B). SmoothAttention computes scales on a per-channel basis within each KV head, matching our pinned models (`Qwen 3 4B`, `Phi-4 mini`, `SmolLM3 3B`, and `Qwen 3 1.7B`).
- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - **Ada Lovelace Architecture Advantage:** Both our local desktop card (RTX 4060, AD107) and cluster card (RTX 4090 capped at 16 GB, AD102) belong to the NVIDIA Ada Lovelace microarchitecture. Table 1 and Figure 15 in the paper prove that Ada Lovelace GPUs (like the L40S) possess significantly higher CUDA core math throughput relative to memory bandwidth compared to Ampere (A100). On Ada Lovelace, even naive KV4 attention was $1.7\times$ faster than KV8, and QServe achieved up to $3.47\times$ throughput speedups. This confirms that 4-bit KV cache kernels will run exceptionally well on our consumer hardware.
  - **VRAM Headroom Calculations (Unmodified BF16 Weights + 4-Bit KV):**
    - *Primary Model (`Qwen 3 4B Instruct`):* Native BF16 weights occupy $\approx 8.06\text{ GB}$, leaving $\approx 7.94\text{ GB}$ of VRAM headroom on our 16 GB cap.
    - *Full BF16 KV Cache:* Consumes $144\text{ KiB/token}$. A single 32K context requires $4.50\text{ GB}$, allowing only 1 user. Ten concurrent 32K sessions would require $45.0\text{ GB}$, instantly causing an Out-Of-Memory (OOM) crash.
    - *With QServe KV4 ($4\times$ Compression):* KV footprint drops from $144\text{ KiB/token}$ to $36\text{ KiB/token}$. A 32K context requires only $1.125\text{ GB}$. Under our $7.94\text{ GB}$ headroom, we can comfortably support **6 to 7 concurrent 32K long-context sessions** or **14 concurrent 8K sessions** on a single 16 GB GPU.
    - *Local 8 GB Card (`Qwen 3 1.7B`):* Weights consume $3.45\text{ GB}$, leaving $4.55\text{ GB}$ headroom. KV4 cuts a 32K context from $3.50\text{ GB}$ to $0.875\text{ GB}$, enabling long-context inference on an 8 GB RTX 4060 without OOM.
- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** **High** if writing bare-metal CUDA/PTX assembly kernels; **Medium** if implementing the SmoothAttention Key/Query transformation in PyTorch hooks and leveraging OpenAI Triton (or vLLM's existing FP8/INT4 PagedAttention kernel infrastructure) for the decode attention loop.
  - **WP-3 Micro-Benchmarking:** Ideal candidate for micro-benchmarking decode attention latency across sequence lengths ($128 \to 32,768$) to empirically verify the CUDA core roofline turning point on our RTX 4060 and RTX 4090 GPUs.
  - **WP-6 Engine Integration (`vLLM` PagedAttention & `LMCache`):**
    - QServe's layout packs scales and zero offsets directly into paged KV blocks, which conforms directly to vLLM's PagedAttention block manager.
    - In `LMCache`, quantizing evicted or low-priority KV chunks to 4 bits cuts host-device PCIe transmission volume by $4\times$, dramatically reducing PCIe transfer stalls during CPU offloading and prefetching.
- **Classification for Project:** **Component Inspiration for WP-4 & Core Blueprint for Selective Quantization** (Provides the mathematical justification for Key outlier smoothing via RoPE commutativity and the engineering rules required to prevent 4-bit dequantization from stalling consumer GPU CUDA cores).
