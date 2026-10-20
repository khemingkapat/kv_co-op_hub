# HeadQ: Model-Visible Distortion and a Prototype KV Quantizer-Corrector Codec

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  When researchers compress the Key-Value (KV) cache of a large language model to 2-bit or 4-bit numbers, they almost always try to minimize storage error—meaning they try to make the compressed numbers in memory look as close as possible to the uncompressed numbers using standard mean-squared error.
  However, this completely misses how attention actually works inside a Transformer:
  1. *Keys are only seen through dot-product scores:* A query vector multiplies a key vector to produce an attention score, which is then fed into a softmax function. If an error in a key points in a direction that the query never looks at, that error is 100% invisible to the model. Even more strikingly, if an error adds the exact same constant offset to all keys, the softmax function cancels it out completely. Standard compression wastes precious bits trying to fix these invisible errors while ignoring tiny errors that directly flip which token gets attended to.
  2. *Values are read through attention-weighted averages:* Unlike keys, values do not determine where the model looks; they determine what information gets gathered. Errors in tokens that receive zero attention are completely harmless, whereas errors in tokens with heavy attention dominate the final output.

- **How HeadQ Solves It (Step-by-Step in Plain Words):**
  1. **Measure What the Model Actually Sees:** The paper defines mathematical metrics for visible distortion rather than storage distortion. For keys, it measures error in score space using attention probability divergence. For values, it measures error in the final readout weighted by squared attention probabilities.
  2. **Base Low-Bit Compression:** It compresses the keys and values using standard low-bit formats (such as 2-bit or 4-bit integers).
  3. **Learn the Query Directions Ahead of Time (Query PCA):** During an offline calibration pass, the authors record the query vectors produced by each attention head. Even though future user queries are unknown, queries within a specific head cluster strongly into a tiny four-dimensional subspace. The author extracts the top four principal directions for each head.
  4. **Store a Tiny Key Side-Code (The Corrector):** When a key is saved to the cache, the method calculates the reconstruction error between the uncompressed key and the 2-bit base key. It projects this residual error onto the four learned query directions and stores just four 4-bit numbers as a tiny side-code (adding only two bytes per token per head).
  5. **Fast Additive Logit Correction at Read Time:** When generating a new token, the model needs the attention score between the current query and past keys. Instead of dequantizing the entire key back to high precision, it computes the coarse score from the 2-bit key and simply adds a quick correction: the query projects into the four-dimensional subspace and takes a four-number dot product with the tiny side-code.
  6. **Squared-Attention Value Policy:** For values, it allocates quantization precision according to how frequently and strongly tokens are attended to, matching the readout operator.

- **How It Compares to Our Project:**
  Our Senior Project explores the three-dimensional co-optimization of KV caches across **Structural** (quantization and token pruning), **Spatial** (VRAM vs. host RAM offloading), and **Temporal** (prefetching and lifecycle reuse). HeadQ is a purely **Structural** advance. It demonstrates that our structural quantization tier should never use raw tensor mean-squared error as an optimization loss. Instead, we can apply its low-rank logit-correction mechanism to protect critical attention routing while holding keys in aggressive 2-bit formats.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** At extreme 2-bit key compression, HeadQ eliminates 84% to 94% of the catastrophic perplexity spike suffered by standard baselines (e.g., cutting excess perplexity from +41.5% down to +6.0% on Qwen2-0.5B, and from +116.9% down to +7.6% on Pythia-160M), while adding only about 3% to 6% extra memory for the side-code. When combined with its value policy into a complete 2-bit full-KV codec, it improves perplexity across all six evaluated models.
  - **What You Pay:** Requires storing and fetching a side-code alongside each key; introduces an extra small dot product during the attention score calculation; currently validated in Python/PyTorch eager hooks without fused CUDA kernels; does not eliminate route-flipping errors in small models when attention distributions are razor-sharp.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Storage MSE is Behaviorally Deceptive:** The paper provides concrete counterexamples where a quantizer with much lower Euclidean key reconstruction error actually causes far worse attention divergence and higher language model perplexity than a quantizer with higher storage error. We must never benchmark our quantization policies on raw tensor reconstruction loss alone.
  2. **Pre-RoPE vs. Post-RoPE Common Mean Cancellation:** Shifting all keys by a constant vector adds an identical offset to all attention scores, which softmax cancels out completely without altering token probabilities. However, fixed quantization grids that do not center keys waste valuable bit dynamic range on this harmless shift, causing precision collapse. Simply centering keys before quantization (or using per-channel scaling) eliminates this failure mode for free.
  3. **Low-Rank Query Subspace Stability:** Even though decode queries change with every newly generated token, the queries within any single attention head remain heavily concentrated in a tiny subspace (rank 4 captures roughly 87% of the query variance). This makes it possible to pre-calculate a static projection basis from calibration data that stays effective across unseen prompts.
  4. **Associative Logit Correction is Computationally Cheap:** Because matrix multiplication is associative, the model does not need to expand the low-rank side-code back into the high-dimensional key vector before computing attention. Instead, the single query vector is projected into the 4-dimensional basis once per head, and that 4-element vector is multiplied by each token's 4-element side-code, costing only four multiply-adds per token.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **No Fused CUDA Serving Kernels:** The paper is strictly an algorithmic proof-of-concept tested via PyTorch eager hooks. There are no fused Triton or CUDA kernels implementing the additive logit correction inside FlashAttention or PagedAttention. Without a custom fused attention kernel, executing separate PyTorch operations during decode on consumer GPUs (RTX 4060 / RTX 4090) would cause substantial kernel launch overhead and kill decode throughput.
  - **Memory Layout Fragmentation:** Storing a separate side-code (4 values at 4 bits = 16 bits) alongside a 2-bit base key vector breaks standard hardware alignment (e.g., 16-byte or 32-byte cache lines), requiring careful bit-packing kernels to prevent memory read amplification over consumer PCIe and VRAM buses.
  - **Small Model Route-Flip Anomaly:** On smaller models with sharp, low-entropy attention distributions (like Pythia-160M), even tiny score errors can flip the top-1 attended token, causing the local quadratic metric to break down.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our manual notes correctly identified the core idea: shifting optimization from storage error to model-visible attention KL divergence, introducing a 2-bit quantizer with a corrector side-code, testing on a laptop RTX 4090 across six models, and noting that the paper is a proof-of-concept without fused serving benchmarks.
  - *Refinements & Corrections from PDF:*
    1. The manual review noted that "PPL retains at 80-90% after quantized." To be precise, the paper reports that HeadQ *eliminates 84% to 94% of the excess perplexity spike* ($\Delta\text{PPL}$) caused by 2-bit key quantization on the most heavily damaged models, rather than achieving an absolute 80–90% retention rate.
    2. The manual review noted that "the side-code that they introduced is addition to storage too." The PDF explicitly proves that this overhead is tiny: rank-4 with 4-bit side codes requires only 16 bits per head per token, which is only a 3.1% to 5.7% storage overhead relative to 4-bit keys.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Jorge L. Ruiz Williams (Independent Researcher)
- **Publication / Venue & Year:** arXiv preprint arXiv:2605.03562v2 [cs.LG], May 8, 2026.
- **Primary Axis:** Structural (Model-visible distortion metric: Key score-space Fisher quadratic & Value $A^2$ readout surrogate; 2-bit quantizer-corrector codec with low-rank query-PCA side code).
- **Secondary Interacting Axis:** Spatial / Precision (Key vs. Value precision asymmetry; 2-bit base quantization with rank-4 side-code; asymmetric treatment of Keys vs. Values).
- **Official Codebase / Repository:** Not released as a public package (tested via PyTorch eager hooks; implementation details and proof-of-concept scripts described in paper).
- **Baseline Implementation Stack:** PyTorch 2.5.1+cu124, Hugging Face Transformers 5.2.0, evaluated on WikiText-103 using GPT-2 124M, Pythia-160M (plus 410M and 1B checkpoints), Qwen2-0.5B, Qwen2.5-0.5B, TinyLlama 1.1B, and Mistral 7B.

---

## 3. How the Algorithm Actually Works (Under the Hood)

```
====================================================================================================
                        HEADQ QUANTIZER-CORRECTOR CODEC WORKFLOW
====================================================================================================

[ Offline Calibration Phase ]
  Collect post-RoPE queries q_h per attention head h
       │
       ▼
  Compute PCA on queries ──► Store top-r orthonormal basis U_h in R^(d x r) (e.g., r = 4)

────────────────────────────────────────────────────────────────────────────────────────────────────

[ Cache Write Path (Key & Value Ingestion) ]
  Uncompressed Key k_j in R^d                        Uncompressed Value v_j in R^d
       │                                                          │
       ├──────────────────────────┐                               ▼
       ▼                          ▼                    [ A^2 Value Policy ]
  Base 2-bit Quantizer      Compute Residual           Quantize based on attention
  k_hat_base = B_K(k_j)     e_j = k_j - k_hat_base     readout weighting
       │                          │                               │
       │                          ▼                               ▼
       │                    Project onto Query Basis          v_hat = B_V(v_j; pi_A^2)
       │                    z_j = Q_q( U_h^T * e_j )              │
       │                    (Store 4 x 4-bit code = 16 bits)      │
       │                          │                               │
       ▼                          ▼                               ▼
  [ Base Key Storage ]     [ Side-Code Storage ]         [ Value Storage ]
  (2-bit VRAM Cache)       (16-bit VRAM Cache)           (2-bit / 4-bit VRAM)

────────────────────────────────────────────────────────────────────────────────────────────────────

[ Cache Read Path (Decode Attention Step for Query q) ]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
  Coarse Key Logit Score                    Low-Rank Additive Correction
  (q^T * k_hat_base) / sqrt(d)              Project Query: q_proj = (q^T * U_h) in R^r
       │                                    Dot Product:   (q_proj * z_tilde_j) / sqrt(d)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
                Corrected Logit s_hat_j = Coarse + Low-Rank Correction
                            │
                            ▼
                Attention Probabilities p = Softmax( s_hat )
                            │
                            ▼
                Attention Readout Output o = Sum_j ( p_j * v_hat_j )
====================================================================================================
```

### Data Structures Added / Modified
- **Learned Query Basis ($U_h$):** A static per-head orthonormal matrix $U_h \in \mathbb{R}^{d \times r}$ representing the top-$r$ principal components of calibration query vectors (default rank $r=4$, head dimension $d \in \{64, 128\}$).
- **Base Quantized KV Caches:** Standard low-bit quantized caches $\hat{K}_{\text{base}} \in \mathbb{R}^{n \times d}$ and $\hat{V}_{\text{base}} \in \mathbb{R}^{n \times d}$ stored at 2, 3, or 4 bits per parameter.
- **Key Residual Side-Code ($Z$):** A compact integer matrix $Z \in \mathbb{Z}^{n \times r}$ holding $q$-bit quantized projections of the key reconstruction error onto $U_h$ (default $q=4$ bits, $r=4$ coordinates $\implies 16$ bits per token per head).

---

### Scoring, Compression & Decision Logic

#### 1. Dense Attention & Score Perturbation
Standard dense scaled dot-product attention computes:
$$s_j = \frac{q^\top k_j}{\sqrt{d}}, \quad p = \text{softmax}(s), \quad o = \sum_{j=1}^n p_j v_j$$
*Plain English Translation:*
- $s_j$: Scalar attention logit score for cached token $j$.
- $q$: Current decode query vector of dimension $d$.
- $k_j$: Cached key vector of dimension $d$ for token position $j$.
- $d$: Head hidden dimension.
- $p$: Vector of attention probabilities across all $n$ cached tokens.
- $o$: Final attention output vector formed by taking the weighted sum of value vectors $v_j$.

When cached keys are quantized, a key perturbation $\delta k_j = \hat{k}_j - k_j$ induces a score perturbation:
$$\delta s_j = \frac{q^\top \delta k_j}{\sqrt{d}}$$
*Plain English Translation:*
- $\delta s_j$: The change in the attention logit for token $j$.
- $\delta k_j$: The error vector between the quantized key and the exact uncompressed key.

---

#### 2. Softmax Fisher Information Geometry
To evaluate how score errors alter the attention probability distribution $p$, the paper expands the Kullback-Leibler (KL) divergence to second order:
$$\text{KL}(p \parallel \hat{p}) = \frac{1}{2} I_{q,K}(\delta K) + \mathcal{O}(\|\delta s\|_2^3)$$
$$I_{q,K}(\delta K) = \delta s^\top \left(\text{diag}(p) - p p^\top\right) \delta s$$
*Plain English Translation:*
- $\text{KL}(p \parallel \hat{p})$: Relative entropy between the exact attention distribution $p$ and the perturbed attention distribution $\hat{p}$.
- $I_{q,K}(\delta K)$: The Fisher information quadratic form measuring model-visible attention distortion.
- $\text{diag}(p) - p p^\top$: The Fisher information matrix of the categorical softmax distribution.

Crucially, this quadratic form simplifies directly to the probability-weighted centered variance of score errors:
$$I_{q,K}(\delta K) = \sum_{j=1}^n p_j (\delta s_j)^2 - \left(\sum_{i=1}^n p_i \delta s_i\right)^2$$
*Plain English Translation:*
- $\sum_{j=1}^n p_j (\delta s_j)^2$: Expected squared score error weighted by token attention probabilities.
- $\left(\sum_{i=1}^n p_i \delta s_i\right)^2$: The squared mean score error across tokens.
- *Core Insight:* Any score shift that is identical across all tokens cancels out completely in this subtraction, proving that constant offsets do not distort attention probabilities.

---

#### 3. Softmax-Null Directions & Visible Quotient Space
The paper proves that key perturbations contain a large subspace that is mathematically invisible to softmax attention.

**Proposition 1 (Softmax-Invariant Key Translation):**
For any query $q$ and any token-shared key translation vector $\mu \in \mathbb{R}^d$:
$$\text{softmax}\left(\frac{q (K + \mathbf{1}\mu^\top)^\top}{\sqrt{d}}\right) = \text{softmax}\left(\frac{q K^\top}{\sqrt{d}}\right)$$
*Plain English Translation:*
- $\mu$: A fixed vector added to every single key in the cache.
- $\mathbf{1}\mu^\top$: An $n \times d$ matrix where every row is $\mu$.
- *Core Insight:* Adding $\mu$ to all keys adds the constant scalar $q^\top \mu / \sqrt{d}$ to every logit. Softmax is invariant to adding a constant to all inputs, so the attention probabilities are exactly identical.

**Theorem 1 (Fixed-Query Active/Null Decomposition):**
Let $\hat{q} = q / \|q\|_2$ be the unit query vector, and let $\bar{\delta} = \frac{1}{n}\sum_{j=1}^n \delta k_j$ be the average key error across all tokens. Any key error row decomposes into three orthogonal pieces:
$$\delta k_j = \bar{\delta} + a_j \hat{q} + \nu_j, \quad \text{where } q^\top \nu_j = 0$$
$$\delta s_j = \frac{q^\top \bar{\delta}}{\sqrt{d}} + \frac{\|q\|_2 a_j}{\sqrt{d}}$$
*Plain English Translation:*
- $\bar{\delta}$: The token-shared average error (completely softmax-null; canceled by softmax).
- $\nu_j$: The component of key error perpendicular to the query vector (invisible to dot-product with $q$).
- $a_j$: The scalar active coefficient projecting along the query direction.
- *Core Insight:* Attention probabilities depend *solely* on the centered active coefficients $a_j$. Storage MSE penalizes all three components ($\bar{\delta}$, $\nu_j$, and $a_j$), which is why storage MSE fails to predict downstream model degradation.

---

#### 4. Value Readout Geometry & The $A^2$ Surrogate
Values are read through a linear weighted blend rather than through logits:
$$\Delta o = \sum_{j=1}^n p_j \Delta v_j$$
$$\frac{1}{d}\|\Delta o\|_2^2 = \frac{1}{d}\sum_{j=1}^n p_j^2 \|\Delta v_j\|_2^2 + \frac{1}{d}\sum_{i \neq j} p_i p_j \Delta v_i^\top \Delta v_j$$
*Plain English Translation:*
- $\Delta o$: The perturbation in the attention layer's output vector.
- $\Delta v_j$: The quantization error vector in cached value $j$.
- $p_j^2 \|\Delta v_j\|_2^2$: The diagonal squared-attention term.
- $p_i p_j \Delta v_i^\top \Delta v_j$: Cross-token error correlation terms.

Assuming cross-token quantization errors are uncorrelated, the cross terms average to zero in expectation, yielding the **$A^2$-weighted surrogate**:
$$\mathbb{E}\left[\frac{1}{d}\|\Delta o\|_2^2\right] = \frac{1}{d}\sum_{j=1}^n p_j^2 \|\Delta v_j\|_2^2$$
*Plain English Translation:*
- *Core Insight:* While key errors matter through the softmax Fisher variance, value errors matter in proportion to the squared attention weight $p_j^2$. Highly attended tokens require high value precision, whereas unattended tokens tolerate extreme value quantization.

---

#### 5. HeadQ Quantizer-Corrector Formulation
HeadQ pairs a base low-bit key quantizer with an additive logit corrector.

**Offline Calibration:**
For each layer and attention head $h$, collect post-RoPE query vectors from calibration text and compute the top-$r$ principal component basis:
$$U_h \in \mathbb{R}^{d \times r}, \quad U_h^\top U_h = I_r$$
*Plain English Translation:*
- $U_h$: Orthonormal matrix whose $r$ columns span the dominant directions that head $h$'s queries explore.

**Cache Write (Key Ingestion):**
When key $k_j$ arrives, compute its base quantization, find the residual, and store a compact side-code:
$$\hat{k}_j^{\text{base}} = \mathcal{B}_K(k_j), \quad e_j = k_j - \hat{k}_j^{\text{base}}, \quad z_j = \mathcal{Q}_q(U_h^\top e_j)$$
*Plain English Translation:*
- $\hat{k}_j^{\text{base}}$: Base low-bit key (e.g., 2-bit integer).
- $e_j$: The uncompressed residual vector of dimension $d$.
- $U_h^\top e_j$: Projection of the residual into the $r$-dimensional query subspace.
- $z_j$: $q$-bit quantized side-code containing $r$ small integers (e.g., $r=4, q=4 \implies 16$ bits total).

**Cache Read (Additive Logit Correction):**
At decode time, when query $q$ attends to token $j$, dequantize the side-code to $\tilde{z}_j$ and correct the logit score:
$$\hat{s}_j = \frac{q^\top \hat{k}_j^{\text{base}}}{\sqrt{d}} + \frac{(q^\top U_h) \tilde{z}_j}{\sqrt{d}}$$
*Plain English Translation:*
- $\frac{q^\top \hat{k}_j^{\text{base}}}{\sqrt{d}}$: The standard low-bit attention score.
- $(q^\top U_h)$: The query vector projected onto the $r$-dimensional basis (computed **once per head**, yielding an $r$-dimensional vector).
- $(q^\top U_h) \tilde{z}_j$: A tiny $r$-element dot product between the projected query and token $j$'s side-code.
- *Core Insight:* Associativity eliminates full key dequantization: $(q^\top U_h) \tilde{z}_j = q^\top (U_h \tilde{z}_j)$. The model only computes $r$ multiply-adds per token!

---

### Memory Hierarchy & Placement
- **Key Base Cache:** Resides in GPU VRAM at 2-bit or 4-bit precision.
- **Key Side-Code Cache:** Resides in GPU VRAM alongside the base cache, adding $rq$ bits per key (16 bits = 2 bytes per key token per head for $r=4, q=4$).
- **Value Cache:** Resides in GPU VRAM, quantized via the $A^2$-weighted policy.
- **Query Basis Storage:** $U_h$ requires $d \times r \times 2\text{ bytes}$ per head (for $d=128, r=4$, this is 1 KB per head, taking less than 32 KB total VRAM across all layers).

---

### Decode-Phase Overheads & Computational Complexity
- **Query Projection:** Computing $q^\top U_h$ requires $d \cdot r$ multiply-adds per head, executed exactly once per decode step. For $d=128, r=4$, this is only 512 FLOPs per head (negligible).
- **Logit Correction:** Computing $(q^\top U_h) \tilde{z}_j$ requires $r$ multiply-adds per cached token per head. For context length $N=4096$ and $r=4$, this adds 16,384 FLOPs per head per decode token.
- **Kernel Overhead:** In standard PyTorch eager mode, dispatching separate correction kernels incurs high latency. In a fused CUDA/Triton implementation, the 4-element dot product can be directly accumulated inside the SRAM-tiled attention loop.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

### Memory Saved (Efficiency)
- **Base Bitwidths Evaluated:** 2-bit, 3-bit, and 4-bit quantization for Keys and Values.
- **Side-Code Overhead Ratio:** Evaluated with rank $r \in \{1, 2, 4, 8\}$ and side-quantization bits $q \in \{2, 4\}$.
  - Default paper setting ($r=4, q=4$): Adds $16\text{ bits}$ ($2\text{ bytes}$) per cached token per head.
  - Relative overhead compared to 4-bit key storage ($d=128$): $\frac{rq}{4d} = \frac{16}{512} = 3.125\%$.
  - Relative overhead for $d=64$ (Pythia / GPT-2): $\frac{16}{256} = 6.25\%$ (empirically measured as $5.73\%$ in Table 3).

---

### Speed & Latency
- **Wall-Clock Decode Speed & Tail Latency:** **Not reported in paper.** The author explicitly notes that the paper is an algorithmic and geometric proof-of-concept using Python/PyTorch eager-attention hooks; fused CUDA kernels and hardware serving latency benchmarks are deferred to future systems work.

---

### Output Quality & Trustworthiness

#### 1. Fisher vs. Storage MSE KL-Ranking Correlation (Table 1)
Spearman rank correlation ($\rho_{\text{KL}}$) between predicted damage metrics and exact attention KL divergence across held-out decode steps on WikiText-103:

| Model | Fisher Proxy | Storage Key MSE | Gain ($\Delta\rho$) |
| :--- | :---: | :---: | :---: |
| **GPT-2 124M** | **0.995** | 0.329 | +0.666 |
| **Pythia-160M** | **0.696** | 0.031 | +0.665 |
| **Qwen2-0.5B** | **0.993** | 0.354 | +0.639 |
| **Qwen2.5-0.5B** | **0.991** | 0.343 | +0.648 |
| **TinyLlama 1.1B** | **0.996** | 0.302 | +0.694 |
| **Mistral 7B** | **0.997** | 0.376 | +0.621 |

*Takeaway:* Raw key MSE completely fails to track attention damage ($\rho \approx 0.03\text{--}0.38$), while the Fisher score-space metric achieves near-perfect correlation ($\rho > 0.99$ on five of six models).

---

#### 2. K-Side Counterexample to Raw Key MSE (Table 2)
Direct comparison between a Fisher-Pullback Quantizer (FPQ) and a Centered TurboQuant-MSE baseline at an identical 3-bit budget:

| Model | Method | Key MSE (Lower=Better) | Fisher RMSE (Lower=Better) | Attention KL (Lower=Better) | Output MSE (Lower=Better) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen2-0.5B** | **FPQ (Score-Visible)** | 0.137461 | **0.20481** | **0.05222** | **0.002205** |
| | Ctrd. TQ-MSE (Storage) | **0.081915** | 0.46641 | 0.40414 | 0.006363 |
| **GPT-2 124M** | **FPQ (Score-Visible)** | 0.131712 | **0.17289** | **0.03480** | **0.002052** |
| | Ctrd. TQ-MSE (Storage) | **0.040432** | 0.21239 | 0.05914 | 0.003535 |

*Takeaway:* Centered TQ-MSE achieves less than one-third the raw storage error on GPT-2 (0.040 vs. 0.132), yet causes almost double the attention KL divergence (0.059 vs. 0.035) and substantially worse output MSE!

---

#### 3. Deployable Query-PCA Approximation Sweep (Table 3)
Evaluation of learned query bases across 12 calibration prompts and 4 evaluation prompts:

| Basis Configuration | Fisher / Base | Attention KL / Base | Key MSE / Base | Query Axis Capture | Storage Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Centered query-PCA, Rank 1 | 0.2801 | 0.5240 | 0.9857 | 0.8144 | 0.0143 |
| **Centered query-PCA, Rank 4 (Default)** | **0.1727** | **0.1920** | **0.9416** | **0.8718** | **0.0573** |
| Centered query-PCA, Rank 8 | 0.1463 | 0.1586 | 0.8880 | 0.8967 | 0.1146 |
| Affine query-PCA, Rank 8 | 0.1396 | 0.1463 | 0.8762 | 0.8967 | 0.1146 |
| Centered Random Basis, Rank 1 | 1.0961 | 1.5035 | 0.9861 | 0.0151 | 0.0143 |

*Takeaway:* Rank-4 query PCA captures 87.2% of query subspace variance and slashes attention KL by 80.8% (down to 0.192 of base), whereas an equal-overhead random basis worsens attention KL by 50.3%.

---

#### 4. Headline Downstream Key-Only Perplexity Deltas (Table 5)
Percent perplexity increase ($\Delta\text{PPL} \text{ \%}$) versus dense FP16 baseline across four WikiText-103 windows (lower is better):

| Model | Key Precision | Baseline Quantizer | HeadQ Rank 2 | HeadQ Rank 4 (Default) | Excess PPL Removed (Rank 4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2-0.5B** | 4-bit | +0.744% | +0.113% | **+0.091%** | 87.8% |
| | 3-bit | +3.073% | +1.697% | **+0.389%** | 87.3% |
| | 2-bit | +41.510% | +11.374% | **+6.022%** | **85.5%** |
| **Pythia-160M** | 4-bit | +3.789% | +0.846% | **+0.701%** | 81.5% |
| | 3-bit | +15.656% | +1.239% | **+1.260%** | 92.0% |
| | 2-bit | +116.894% | +9.165% | **+7.631%** | **93.5%** |
| **GPT-2 124M** | 4-bit | +0.859% | +0.131% | **+0.099%** | 88.5% |
| | 3-bit | -0.120% | +0.798% | +0.628% | (Noise floor) |
| | 2-bit | +15.630% | +2.976% | **+1.705%** | **89.1%** |
| **Qwen2.5-0.5B**| 4-bit | +0.110% | -0.121% | **-0.583%** | (Near dense) |
| | 3-bit | +1.775% | -0.310% | **-0.158%** | 100.0% |
| | 2-bit | +49.135% | +9.157% | **+7.846%** | **84.0%** |
| **TinyLlama 1.1B**| 4-bit | -0.090% | -0.112% | **-0.234%** | (Near dense) |
| | 3-bit | +1.379% | +0.677% | **+0.178%** | 87.1% |
| | 2-bit | +15.797% | +12.737% | **+6.729%** | **57.4%** |
| **Mistral 7B** | 4-bit | +0.232% | +0.256% | **+0.132%** | (Near dense) |
| | 3-bit | -0.036% | +0.258% | **-0.090%** | (Near dense) |
| | 2-bit | +2.140% | +2.240% | +2.270% | (Near dense) |

---

#### 5. Full-KV Codec Composition Perplexity Deltas (Table 7)
Composing HeadQ on keys with the $A^2$-guided value allocator on values across both keys and values:

| Model | KV Bits | Base Quantizer $\Delta\text{PPL}$ | Full Codec $\Delta\text{PPL}$ (HeadQ + $A^2$ V) | Relative PPL Rescue |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2-0.5B** | 2-bit | +84.867% | **+23.108%** | **-61.76%** |
| **Pythia-160M** | 2-bit | +173.704% | **+19.795%** | **-153.91%** |
| **GPT-2 124M** | 2-bit | +24.634% | **+9.072%** | **-15.56%** |
| **Qwen2.5-0.5B**| 2-bit | +94.700% | **+24.498%** | **-70.20%** |
| **TinyLlama 1.1B**| 2-bit | +25.248% | **+14.562%** | **-10.69%** |
| **Mistral 7B** | 2-bit | +7.278% | **+2.507%** | **-4.77%** |

*Takeaway:* The composed codec rescues 2-bit performance on all six models without exception, proving that the key-side score-space corrector and value-side readout surrogate operate synergistically.

---

#### 6. Deployed Sign Control Check (Table 6)
Evaluating HeadQ with the corrector sign reversed at 3 key bits (rank 4, 4 side bits):
- **Qwen2-0.5B:** Base = $+3.073\%$, HeadQ = $+0.278\%$, Wrong-Sign HeadQ = $+1.716\%$.
- **Pythia-160M:** Base = $+15.656\%$, HeadQ = $-0.328\%$, Wrong-Sign HeadQ = **$+51.888\%$** (catastrophic breakdown).
*Takeaway:* The performance gain is strictly causal—it stems from subtracting the error in the active query direction, not from adding arbitrary side capacity.

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16 Weights Rule:** Fully compliant. Model weights remain completely untouched in native BF16 precision; all operations target cached KV tensors exclusively.
  - **Dense GQA Compatibility:** All primary models in our benchmark suite (**Qwen 3 4B**, **Phi-4 mini**, **Ministral 3 3B**, **SmolLM3 3B**, **Qwen 3 1.7B**) utilize Grouped Query Attention (GQA). In GQA, multiple query heads share a single key-value head. HeadQ is compatible with GQA in two ways:
    1. *Group-Pooled Query Basis:* We can compute a single shared PCA basis $U_k \in \mathbb{R}^{d \times r}$ by pooling calibration queries across all query heads that attend to key-value head $k$.
    2. *Per-Query Head Projection:* At decode time, each individual query head $q_h$ still computes its own projected vector $q_h^\top U_k \in \mathbb{R}^r$ before taking the dot product with token side-codes $\tilde{z}_j$.

- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - **Memory Footprint:** For an 8K context on a 4B model (e.g., Qwen 3 4B with 36 layers, 4 KV heads, head dimension 128):
    - Full FP16 KV cache: $\approx 1.89\text{ GB}$.
    - 2-bit base cache: $\approx 0.24\text{ GB}$.
    - HeadQ rank-4 side-code ($16\text{ bits}$ per token): $\approx 0.12\text{ GB}$.
    - Total HeadQ cache: $\approx 0.36\text{ GB}$ (an $81\%$ reduction over FP16), fitting comfortably within the 8 GB RTX 4060 alongside native BF16 model weights ($\approx 7.5\text{ GB}$ total VRAM).
  - **Host RAM Requirements:** Storing pre-computed PCA basis matrices across 36 layers requires less than 2 MB of host RAM.

- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** High for serving integration; Medium for evaluation microbenchmarking.
  - **WP-3 Micro-Benchmarking & Metric Selection:**
    - *Immediate Adoption:* We should adopt the **Softmax Fisher Information Quadratic** (Equation 6) and **$A^2$-Weighted Value Readout Error** (Equation 14) as our primary proxy loss functions in WP-3 when comparing eviction and tiered quantization schemes. Raw tensor MSE should be permanently deprecated as a proxy.
  - **WP-6 Engine Integration (`vLLM` PagedAttention / `LMCache`):**
    - *Challenge:* vLLM's PagedAttention kernel assumes standard dot-product attention over contiguous or paged key blocks. Injecting an additive side-code correction $\frac{(q^\top U_h) \tilde{z}_j}{\sqrt{d}}$ requires either:
      1. Modifying the PagedAttention Triton/CUDA kernel to accept an additional `side_code_ptr` and accumulate the 4-element dot product before softmax; or
      2. Applying the correction as an external logit bias before softmax, which breaks FlashAttention/PagedAttention fusion and incurs kernel launch stalls.
    - *Feasible Adaptation:* In LMCache, we can use HeadQ's low-rank residual side-code format to compress cached blocks when transferring them across PCIe to host RAM. During CPU-to-GPU retrieval, keys can either be restored to 4-bit/BF16 or served using a specialized Triton decode kernel.

- **Classification for Project:** **Component Inspiration & Analytical Grounding** (Provides the foundational mathematical proof that Key and Value errors must be measured in model-visible coordinates, and demonstrates an ultra-low-rank logit-correction mechanism for extreme 2-bit key compression).
