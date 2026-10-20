# TurboQuant & Statistical Quality Measures: Statistical Inference and Quality Measures of KV Cache Quantisations

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  In large language models, the Key-Value (KV) cache grows with every new token, quickly exhausting GPU memory during long conversations or multi-user serving. Most engineers compress both Keys and Values symmetrically (for example, shrinking both to 4-bit integers using standard mean squared error). However, attention treats Keys and Values completely differently:
  1. *Keys determine routing (who gets attention):* Keys pass through an exponential softmax function. If a Key is slightly corrupted, the exponential function magnifies that error, potentially picking the completely wrong token and causing gibberish outputs.
  2. *Values determine the content blend (what information is gathered):* Values are simply blended together through a weighted average. Small, zero-mean errors across many tokens naturally average out and cancel each other.
  Treating them identically wastes memory precision on Values while starving Keys of the directional fidelity they desperately need. Furthermore, standard scalar error metrics measured on memory storage completely fail to predict whether the model's generated text will survive or collapse.

- **How TurboQuant & This Paper Solve It (Step-by-Step in Plain Words):**
  1. **Disperse Outliers via Random Rotation (Stage 1):** Before quantizing, the cache vector is rotated using an orthonormal Walsh-Hadamard butterfly transform combined with random sign flips. This spreads out spiky outlier values evenly across all dimensions, transforming arbitrary token embeddings into a mathematically predictable Beta distribution on the unit sphere. Each coordinate is then compressed using a scalar Lloyd-Max codebook specifically tuned for this Beta shape.
  2. **Clean Up Residuals with a 1-Bit Sketch (Stage 2):** To correct remaining errors without storing bulky high-precision numbers, a 1-bit Quantized Johnson-Lindenstrauss (QJL) random sign sketch is saved alongside the quantized residual, scaled by a theoretical constant to guarantee unbiased dot products.
  3. **The Critical Asymmetry Discovery (KQV vs. QKQV):** The paper proves that applying this 1-bit residual sketch to *Values* is beneficial, but applying it to *Keys* is counterproductive. Trading away a single scalar bit on Keys to pay for a 1-bit sign sketch inflates the variance of dot products by more than six-fold. The exponential softmax then acts like a megaphone, blowing up this variance and corrupting attention routing. Therefore, the optimal architecture (named KQV) uses pure rotated scalar quantization on Keys, and scalar quantization plus 1-bit residual sketching on Values.
  4. **The Six-Dimensional Quality Framework:** Instead of relying on single average error numbers that hide catastrophic failures, the author decomposes errors into two orthogonal components (scale error and directional angle error) across three critical locations: the Key cache, the Value cache, and the final attention output.

- **How It Compares to Our Project:**
  Our project explores co-optimizing KV caches across Structural (quantization and token eviction), Spatial (CPU RAM offloading), and Temporal (prefetching and transfer scheduling) dimensions. This paper provides the mathematical and empirical justification for our Structural design: we should never quantize Keys and Values symmetrically. Keys require higher precision and exact directional alignment to maintain routing integrity, whereas Values can be compressed far more aggressively because their linear combination naturally absorbs zero-mean noise.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** At the industry-standard 4-bit budget, the asymmetric KQV scheme achieves over two times lower attention divergence than symmetric approaches, preserves top-token retrieval accuracy, and completely neutralizes heavy-tailed outlier tokens across extreme distributions.
  - **What You Pay:** Rotating vectors requires a fast butterfly permutation step during cache writes and reads. Furthermore, when Keys share a low-rank subspace (common in real transformers), the rotation disguises the low-rank structure in coordinate histograms but still suffers severe output degradation when attention concentrates on dominant tokens.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Strictly Asymmetric Precision for Keys vs. Values:** We should allocate our memory budget unevenly. For instance, allocating 4-bit or 5-bit scalar precision to Keys while giving Values 2-bit or 3-bit scalar storage combined with residual correction yields significantly better end-to-end generation quality than forcing both to 4 bits.
  2. **The 4-Bit Deployment Sweet Spot:** At 4 bits per vector coordinate—the exact target for practical consumer GPU deployment—the asymmetric KQV scheme universally beats both naive scalar compression and symmetric residual sketching across every statistical metric, all tested ranks, and all tail distributions.
  3. **Walsh-Hadamard Rotation as a Distribution Normalizer:** A structured butterfly rotation requires zero learnable parameters, runs in near-linear time, and reliably converts spiky, heavy-tailed embedding vectors into uniform Beta distributions. This allows us to use static, pre-computed quantization tables rather than calculating dynamic per-token min-max grids on the fly.
  4. **Decomposing Storage Error from Directional Routing Error:** Never evaluate a compression scheme using only mean squared error on disk or VRAM. A scheme can achieve decent mean squared error while slightly rotating a Key vector, which completely redirects the attention softmax to an irrelevant token. Measuring directional cosine distance at the Key level directly predicts whether attention routing will collapse.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **The Low-Rank Subspace Trap:** Real transformer attention matrices are intrinsically low-rank. While the Hadamard rotation makes individual coordinates appear perfectly Gaussian or Beta-distributed, it preserves the underlying geometric subspace alignment. When a query strongly aligns with dominant subspace keys, the softmax exponentially magnifies tiny quantization imperfections. We cannot rely on rotation alone to solve low-rank bottlenecks; selective preservation of dominant singular directions or rank-aware codebooks is necessary.
  - **Kernel Fusion Overhead on Consumer GPUs:** On memory-bandwidth-constrained desktop cards (such as an RTX 4060 with 272 gigabytes per second of bandwidth), dequantization and butterfly transforms must be completely fused into the attention read loop. If the rotation and residual addition require separate CUDA kernel launches or extra round-trips through global video memory, the latency penalty will erase all memory bandwidth benefits.
  - **Argmax Shift on Focused Needle Queries:** When a query targets a single specific token (as in passkey retrieval or needle-in-a-haystack tasks), any small directional error in that dominant Key can shift the top attention winner to the wrong token, causing complete retrieval failure.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our manual review notes (Row 31) correctly identified the primary message: Key and Value caches exhibit strong mathematical asymmetry, scalar storage error poorly predicts model-visible attention logit distortion, and Key distortion causes catastrophic routing collapse while Value distortion merely perturbs the linear blend.
  - *Refinements & Additions from PDF:*
    1. The paper evaluates three precise formal schemes: `KV` (untransformed scalar Lloyd-Max baseline), `KQV` (WHT + scalar on K; WHT + scalar + QJL on V), and `QKQV` (WHT + scalar + QJL on both).
    2. Trading a scalar bit for a 1-bit QJL sketch on Keys mathematically inflates inner product variance by a factor of 2 pi (roughly 6.28 times), explaining why QJL on Keys is fundamentally flawed under a fair bit budget.
    3. The paper reveals a budget-dependent crossover: QKQV achieves better geometric Key reconstruction at odd bit depths (2, 3, 5, 7 bits), while KQV wins at even bit depths (4 and 6 bits). However, at the critical 4-bit budget, KQV wins decisively across all metrics.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Paolo D'Alberto (AMD, `paolodalberto@amd.com`), in collaboration with Claude (Anthropic).
- **Publication / Venue & Year:** arXiv preprint `arXiv:2605.08114v1 [cs.LG]`, published April 27, 2026.
- **Primary Axis:** Structural (Quantization quality measures, asymmetric Key-Value precision allocation, Walsh-Hadamard rotation, Beta Lloyd-Max quantization, and Quantized Johnson-Lindenstrauss residual sketching).
- **Secondary Interacting Axis:** Numerical Fidelity & Statistical Inference (Shannon-$\phi$ plane error decomposition, 6D error vectors, Energy Distance, and Jensen bias propagation in attention softmax).
- **Official Codebase / Repository:** Implemented in AMD ROCm HIP GPU kernels with Python simulation post-processing (detailed in Section 9 of the paper).
- **Baseline Implementation Stack:** Evaluated across synthetic and empirical embedding distributions (Gaussian, fat-tailed Student-$t$, low-rank subspace projections, and focused query configurations) across dimensions $d \in \{2, 8, 128, 1024\}$ and bit budgets $n \in \{2, 3, 4, 5, 6, 7\}$.

---

## 3. How the Algorithm Actually Works (Under the Hood)

### 3.1 Mathematical Foundations: Beta Distribution on the Hypersphere

When a $d$-dimensional unit vector $x$ is uniformly distributed on the sphere $\mathbb{S}^{d-1}$, each individual coordinate $x_j$ follows a Beta distribution mapped to the interval $[-1, 1]$. Its probability density function is:

$$f_X(t) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)} (1 - t^2)^{\frac{d-3}{2}}, \quad t \in [-1, 1]$$

*Plain-English translation:*
- $f_X(t)$ is the probability density of observing value $t$ at any coordinate of a random unit vector on a $d$-dimensional sphere.
- $d$ is the vector embedding dimension (e.g., $d=128$ for transformer attention heads).
- $\Gamma(\cdot)$ is the standard Gamma function governing sphere surface area normalization.
- $(1 - t^2)^{\frac{d-3}{2}}$ describes the shrinking surface area of a $(d-2)$-dimensional cross-sectional slice as the coordinate moves from the equator ($t=0$) toward the poles ($t=\pm 1$).

As $d \to \infty$, this density converges asymptotically to a zero-mean Gaussian distribution with variance $1/d$:

$$\lim_{d \to \infty} (1 - t^2)^{\frac{d-3}{2}} \approx e^{-\frac{d t^2}{2}} \sim \mathcal{N}\left(0, \frac{1}{d}\right)$$

*Plain-English translation:*
- In low dimensions (e.g., $d=2$), the density is U-shaped (an arcsine distribution with coordinates clustering near $\pm 1$).
- In high dimensions (e.g., $d \ge 128$), the coordinates concentrate tightly around zero in a bell shape that closely resembles a Gaussian, but remains strictly bounded within $[-1, 1]$.

### 3.2 The Two-Stage Compression Pipeline

TurboQuant compresses normalized cache vectors through a two-stage pipeline:

```
[ Input Vector v ] 
       │
       ▼
 [ Normalization ] ────> Extract Norm ||v|| (stored as FP32)
       │
       ▼ Unit Vector v̂
[ WHT + Sign Flip ] ───> x_rot = (1 / sqrt(d)) * H * diag(s) * v̂
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ Stage 1: Scalar Quantization ]        [ Stage 2: QJL Sketch (V only) ]
  Beta Lloyd-Max at (n-1) or n bits        Residual r = x_rot - x̂_β
  Yields x̂_β                               q = sign(H * (s' ⊙ r)) ∈ {-1, +1}^d
                                           Yields 1-bit sketch q
```

#### Stage 1: Orthonormal Rotation & Beta-Optimal Scalar Quantization
To eliminate coordinate outliers and axis alignment, the unit vector $\hat{v} = v / \|v\|$ is rotated:

$$x_{\text{rot}} = \frac{1}{\sqrt{d}} H \operatorname{diag}(s) \hat{v}$$

*Plain-English translation:*
- $H \in \{-1, +1\}^{d \times d}$ is an unnormalized Walsh-Hadamard matrix where $H^T H = d I$.
- $\operatorname{diag}(s)$ is a diagonal matrix containing an independent random sign vector $s \in \{-1, +1\}^d$ (a Rademacher sequence).
- $1/\sqrt{d}$ ensures orthonormal energy preservation so that $\|x_{\text{rot}}\| = \|\hat{v}\| = 1$.

Because $x_{\text{rot}}$ coordinates adhere to the Beta distribution, a static scalar Lloyd-Max quantizer designed specifically for this density quantizes each coordinate into $B$ bits, yielding reconstructed coordinates $\hat{x}_\beta$. The vector norm $\|v\|$ is stored separately in full 32-bit floating point precision.

Exact inversion of Stage 1 (without QJL) is mathematically lossless in the transformation itself:

$$\hat{x} = \frac{1}{d} \operatorname{diag}(s) H \operatorname{dequant}(H \operatorname{diag}(s) \hat{v})$$

*Plain-English translation:*
- Because $\operatorname{diag}(s)^2 = I$ and $H^T H = d I$, applying the inverse Hadamard transform and sign un-flipping perfectly reconstructs the original vector up to scalar quantization noise, introducing zero geometric transformation distortion.

#### Stage 2: Quantized Johnson-Lindenstrauss (QJL) Residual Correction
Let $r = x_{\text{rot}} - \hat{x}_\beta$ denote the quantization residual from Stage 1. A 1-bit sign sketch is computed using a freshly sampled independent random sign vector $s' \in \{-1, +1\}^d$:

$$q = \operatorname{sign}\left(H (s' \odot r)\right) \in \{-1, +1\}^d$$

The unbiased reconstructed residual estimator is:

$$\hat{r} = \sqrt{\frac{\pi}{2}} \frac{\|r\|}{d} s' \odot H q, \quad \mathbb{E}[\langle q_{\text{query}}, \hat{r}\rangle] = \langle q_{\text{query}}, r\rangle$$

*Plain-English translation:*
- $q$ is a $d$-dimensional vector of binary bits ($\pm 1$), requiring exactly 1 bit per coordinate of storage.
- $\sqrt{\pi/2} \approx 1.2533$ is the analytic bias correction factor that compensates for the fact that taking the sign of a zero-mean Gaussian variable underestimates its expected magnitude: for $z \sim \mathcal{N}(0, \sigma^2)$, $\mathbb{E}[|z|] = \sigma \sqrt{2/\pi}$.
- $\|r\|$ is the scalar norm of the residual vector, stored as a floating-point scalar.
- This estimator is strictly unbiased: in mathematical expectation over the random sign vectors, the estimated inner product equals the true inner product.

### 3.3 The Core Theoretical Proof: Why QJL on Keys Harms Attention

#### Derivation of the $2\pi$ Inner Product Variance Penalty
In transformer self-attention, queries interact with Keys exclusively through inner products $\langle q, k \rangle$. Consider quantizing a Key vector $k$ under an equal total bit budget of $B$ bits per coordinate:

**Case A: $B$-bit pure scalar Lloyd-Max quantization.**  
Each additional bit in a Lloyd-Max quantizer roughly halves the root mean squared error (the standard $6\text{ dB}$ per bit rule). Thus, dropping one bit quadruples the residual energy:

$$\|r_{B-1}\|^2 \approx 4 \|r_B\|^2$$

For $B$-bit scalar quantization, the reconstructed key is $\hat{k} = k + r_B$. The inner product error $\delta = \langle q, \hat{k}\rangle - \langle q, k\rangle = \langle q, r_B\rangle$ has variance:

$$\operatorname{Var}[\delta]_{\text{scalar}} = \frac{\|q\|^2 \|r_B\|^2}{d}$$

*Plain-English translation:*
- The inner product variance under pure scalar quantization is proportional to the residual error energy $\|r_B\|^2$ divided by dimension $d$.

**Case B: $(B-1)$-bit scalar quantization + 1-bit QJL residual sketch.**  
If we sacrifice 1 scalar bit to store a 1-bit QJL sketch, the residual has energy $\|r_{B-1}\|^2 \approx 4 \|r_B\|^2$. The 1-bit QJL unbiased estimator introduces variance:

$$\operatorname{Var}[\delta]_{\text{QJL}} = \frac{\pi}{2} \frac{\|q\|^2 \|r_{B-1}\|^2}{d} = \frac{\pi}{2} \frac{\|q\|^2 (4 \|r_B\|^2)}{d} = \frac{2\pi \|q\|^2 \|r_B\|^2}{d}$$

*Plain-English translation:*
- The $\pi/2$ factor comes from the sign projection variance. Multiplying by the $4\times$ energy penalty from the lost scalar bit produces a total variance factor of $2\pi$.

**The Fundamental Ratio:**

$$\frac{\operatorname{Var}[\delta]_{\text{QJL}}}{\operatorname{Var}[\delta]_{\text{scalar}}} = 2\pi \approx 6.283$$

*Plain-English translation:*
- Under a fixed bit budget, replacing one bit of scalar precision on Keys with a 1-bit QJL sketch **increases inner product variance by a factor of $2\pi$ (over $6.28\times$)**, completely independent of embedding dimension $d$.

#### Nonlinear Softmax Amplification via Jensen's Inequality
Why is this $2\pi$ variance inflation lethal for Keys but harmless for Values?

1. **Values Propagate Linearly:**
   The Value cache enters the output through a linear combination:
   
   $$T = \sum_{i=1}^S w_i \hat{v}_i = \sum_{i=1}^S w_i (v_i + \delta_{v, i})$$
   
   Across a sequence of length $S$, zero-mean unbiased errors cancel out at the central limit rate of $1/\sqrt{S}$. Because there is no nonlinear transformation, QJL's unbiasedness is genuinely beneficial for Values.

2. **Keys Propagate Nonlinearly Through Softmax:**
   Attention weights are calculated via normalized exponentials:
   
   $$w_i = \frac{e^{\langle q, \hat{k}_i \rangle / \sqrt{d}}}{\sum_j e^{\langle q, \hat{k}_j \rangle / \sqrt{d}}}$$
   
   By Jensen's inequality, because the exponential function $e^x$ is strictly convex, even a zero-mean perturbation $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$ systematically biases the expected numerator upwards:
   
   $$\mathbb{E}\left[e^{\langle q, k_i \rangle + \epsilon_i}\right] = e^{\langle q, k_i \rangle} \cdot e^{\frac{\sigma_i^2}{2}}$$
   
   *Plain-English translation:*
   - Every Key's attention score is inflated by a multiplicative factor $e^{\sigma_i^2 / 2}$.
   - If the error variance $\sigma_i^2$ is non-uniform across keys—which is always true under finite quantization—the dominant tokens with higher error variance receive disproportionate, unearned attention weight.
   - The $2\pi$ variance inflation from QJL drastically increases this distortion.

3. **The Softmax Argmax-Shift Catastrophe:**
   In focused attention (e.g., retrieving a specific fact), a single dominant key $k^* = \arg\max_i \langle q, k_i \rangle$ receives nearly $100\%$ of the attention weight. Inflating inner product variance increases the probability that a random non-target key exceeds the score of $k^*$. If an argmax-shift occurs, the model attends to the wrong token entirely, leading to hallucination or output collapse.

### 3.4 The Shannon-$\phi$ Plane and 6D Error Decomposition

To eliminate the ambiguities of single scalar error metrics, the paper establishes the Shannon-$\phi$ plane, decomposing vector distortion into two canonical, mutually orthogonal axes:

$$\epsilon^{\text{snr}} = \frac{\|x - \hat{x}\|^2}{\|x\|^2} \quad \text{(Scale / Energy Error)}$$

$$\epsilon^{\text{dir}} = 1 - \frac{\langle x, \hat{x}\rangle}{\|x\| \|\hat{x}\|} \quad \text{(Direction / Angular Error, } \epsilon^{\text{dir}} \in [0, 2]\text{)}$$

*Plain-English translation:*
- $\epsilon^{\text{snr}}$ measures Euclidean energy loss (relative mean squared error).
- $\epsilon^{\text{dir}}$ measures cosine angular divergence ($0$ means collinear, $1$ means orthogonal, $2$ means opposite direction).
- These two measures are strictly orthogonal: a vector can have perfect angle but terrible scale, or perfect scale but terrible angle.

This decomposition is measured at three distinct operational sites: the Key cache ($K$), the Value cache ($V$), and the post-softmax attention output ($T = \operatorname{softmax}(Q K^T / \sqrt{d}) V$), yielding the **6D Error Vector**:

$$e = \left(\left[\epsilon^{\text{snr}}_K, \epsilon^{\text{dir}}_K\right], \left[\epsilon^{\text{snr}}_V, \epsilon^{\text{dir}}_V\right], \left[\epsilon^{\text{snr}}_T, \epsilon^{\text{dir}}_T\right]\right)$$

To evaluate statistical significance across multidimensional trial distributions without parametric assumptions, the paper employs the **Energy Distance**:

$$\mathcal{E}(X, Y) = 2 \mathbb{E}[\|x - y\|] - \mathbb{E}\left[\|x - x'\|\right] - \mathbb{E}\left[\|y - y'\|\right]$$

combined with the non-parametric **Mann-Whitney rank-biserial correlation** $r \in [-1, +1]$ and the **Kolmogorov-Smirnov maximum CDF statistic** $D$.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

### 4.1 Scheme Definitions under Fair Bit Budgets

All comparisons enforce strict equality in total stored bit budgets per coordinate:

| Scheme Name | Key Cache ($K$) | Value Cache ($V$) | Walsh-Hadamard Transform | Theoretical Basis |
| :--- | :--- | :--- | :--- | :--- |
| **`KV`** | Scalar Lloyd-Max ($n$ bits) | Scalar Lloyd-Max ($n$ bits) | **No** (Raw untransformed coordinates) | Uncompressed scalar baseline |
| **`KQV`** | WHT + Scalar Lloyd-Max ($n$ bits) | WHT + Scalar ($n-1$ bits) + 1-bit QJL | **Yes** (Applied to both $K$ and $V$) | **Asymmetric design** (TurboQuant optimal) |
| **`QKQV`** | WHT + Scalar ($n-1$ bits) + 1-bit QJL | WHT + Scalar ($n-1$ bits) + 1-bit QJL | **Yes** (Applied to both $K$ and $V$) | **Symmetric design** (QJL applied to both) |

*Note:* All three schemes store the exact vector norm as an FP32 scalar ($4\text{ bytes}$).

### 4.2 Decisive Evidence on Asymmetric Quality (`KQV` vs. `QKQV`)

Table 2 and Table 3 from the paper directly isolate the effect of applying QJL to Keys while holding Value compression strictly identical.

#### Geometric Errors & Output Quality across Distributions (Table 2 Extract)
Evaluated at head dimension $d=128$, comparing median directional errors ($\epsilon^{\text{dir}}$) and Mann-Whitney $r$ on the full 6D distance $d_6$ ($r > 0$ means `KQV` wins; lower $\epsilon$ is better):

| Distribution Mode | Budget ($n$) | KQV $\epsilon^{\text{dir}}_K$ | QKQV $\epsilon^{\text{dir}}_K$ | Winner on $K$ Geometry | KQV $\epsilon^{\text{dir}}_T$ | QKQV $\epsilon^{\text{dir}}_T$ | MW $r$ ($d_6$) | Dominant Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fat-Tail ($\nu=3$)** | $n=2$ | $0.162$ | **$0.065$** | QKQV | $0.306$ | $0.189$ | $-1.000$ | QKQV ($p < 0.005$) |
| | $n=3$ | $0.076$ | **$0.073$** | QKQV | $0.248$ | $0.231$ | $-0.774$ | QKQV ($p < 0.005$) |
| | **$n=4$** | **$0.027$** | $0.032$ | **KQV** | **$0.102$** | $0.138$ | **$+0.984$** | **KQV ($p < 0.005$)** |
| | $n=5$ | $0.021$ | **$0.010$** | QKQV | $0.073$ | $0.048$ | $-1.000$ | QKQV ($p < 0.005$) |
| | **$n=6$** | **$0.003$** | $0.005$ | **KQV** | **$0.019$** | $0.023$ | **$+1.000$** | **KQV ($p < 0.005$)** |
| **Low-Rank ($r=16$)** | $n=3$ | $0.077$ | **$0.075$** | QKQV | $0.145$ | $0.148$ | $-0.449$ | QKQV |
| | **$n=4$** | **$0.027$** | $0.032$ | **KQV** | **$0.075$** | $0.076$ | **$+0.627$** | **KQV ($p < 0.005$)** |
| | $n=5$ | $0.024$ | **$0.010$** | QKQV | $0.093$ | $0.027$ | $-1.000$ | QKQV |
| | **$n=6$** | **$0.003$** | $0.006$ | **KQV** | **$0.019$** | $0.025$ | **$+0.993$** | **KQV ($p < 0.005$)** |

#### Softmax Attention KL Divergence and Routing Recall (Table 3 Extract)
On Fat-Tail inputs ($\nu=3$), measuring median KL divergence and top-5 token routing error ($1 - \text{topk}_5$):

| Budget ($n$) | Median $\text{KL}_{\text{QKQV}}$ | Median $\text{KL}_{\text{KQV}}$ | KL Reduction Factor | Top-5 Recall ($\text{topk}_5$) | MW $r$ on Routing Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $n=2$ | $0.318$ | $0.905$ | $0.35\times$ (QKQV wins) | QKQV higher | $+0.869$ (QKQV wins) |
| $n=3$ | $0.453$ | $0.497$ | $0.91\times$ (QKQV wins) | KQV higher | $-0.397$ (KQV wins) |
| **$n=4$** | **$0.377$** | **$0.167$** | **$2.25\times$ lower KL for KQV** | **$0.831$ (KQV) vs. $0.789$ (QKQV)** | **$-0.983$ ($p < 10^{-23}$, KQV wins)** |
| $n=5$ | $0.067$ | $0.134$ | $0.50\times$ (QKQV wins) | QKQV higher | $+0.992$ (QKQV wins) |
| **$n=6$** | **$0.029$** | **$0.015$** | **$1.93\times$ lower KL for KQV** | **KQV higher** | **$-0.970$ ($p < 0.005$, KQV wins)** |

*Critical Insight:* At the critical $4$-bit deployment budget ($n=4$), **`KQV` beats `QKQV` on every single metric**:
- Median KL divergence is slashed by $2.25\times$ ($0.167$ vs. $0.377$).
- Top-5 routing recall reaches $83.1\%$ vs. $78.9\%$ ($p < 10^{-23}$).
- Output direction error $\epsilon^{\text{dir}}_T$ is $26\%$ lower ($0.102$ vs. $0.138$).

### 4.3 Robustness to Heavy-Tailed Outliers ($\nu$ Parameter Sweep at $n=4$)

Sweeping the Student-$t$ degrees of freedom $\nu$ from $10$ (nearly Gaussian) down to $1.05$ (extreme near-Cauchy heavy tails) shows WHT's remarkable distribution-normalizing capability:

| Tail Heaviness ($\nu$) | `KQV` Direction Error ($\epsilon^{\text{dir}}_K$) | Plain `KV` Direction Error ($\epsilon^{\text{dir}}_K$) | Relative Error Degradation | Energy Distance $\mathcal{E}_{6d}$ |
| :--- | :--- | :--- | :--- | :--- |
| $\nu = 10.0$ (Near-Gaussian) | **$0.0271$** | $0.0294$ | $1.08\times$ higher for KV | $0.063$ ($r = +0.307$) |
| $\nu = 3.0$ | **$0.0271$** | $0.0591$ | $2.18\times$ higher for KV | $0.715$ ($r = +1.000$) |
| $\nu = 2.0$ | **$0.0271$** | $0.0991$ | $3.66\times$ higher for KV | $0.713$ ($r = +1.000$) |
| $\nu = 1.5$ | **$0.0271$** | $0.1419$ | $5.24\times$ higher for KV | $0.902$ ($r = +1.000$) |
| $\nu = 1.05$ (Near-Cauchy) | **$0.0268$** | **$0.2015$** | **$7.52\times$ higher for KV** | **$1.004$ ($r = +1.000$)** |

`KQV` maintains $\epsilon^{\text{dir}}_K \approx 0.027$ invariant to three decimal places across the entire spectrum. In contrast, untransformed `KV` degrades by over $7.5\times$, suffering complete codebook saturation at $\nu \le 1.5$.

### 4.4 The Low-Rank Pathology at High Dimension ($d=1024$)

When vectors lie in a low-rank subspace (rank $r = d/8$):
- **Marginal coordinate histograms:** At $d=1024$, the rotated coordinates of Gaussian, heavy-tailed, and low-rank tensors all produce identical empirical histograms matching $\text{Beta}(511.5, 511.5)$.
- **Softmax KL Divergence Collapse:** Despite the identical coordinate distributions, low-rank inputs cause catastrophic KL divergence:
  - Heavy-tail maximum KL: $\sim 10^{-1}$ nats.
  - Low-rank maximum KL: $\sim 10^0\text{--}10^1$ nats (one to two orders of magnitude worse).
- **Physical Cause:** WHT rotates individual coordinates into a Beta marginal shape, but preserves the internal joint subspace correlation. When queries align within that subspace, attention concentrates into sharp peaks, exponentially magnifying residual direction errors.

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

### 5.1 Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md))

- **Unmodified BF16 Model Weights Rule:** Fully compliant. The TurboQuant formulation leaves transformer model weights completely untouched in native Bfloat16 precision, operating solely on the dynamic runtime KV activations.
- **Dense Grouped-Query Attention (GQA) Compatibility:**
  Our primary pinned models in `docs/MODELS.md` use standard head dimension $D_{\text{head}} = 128$:
  - **Qwen 3 4B Instruct:** $L=36$, $H_{KV}=8$, $D_{\text{head}}=128$ ($144\text{ KiB/token}$).
  - **Phi-4 mini Instruct:** $L=32$, $H_{KV}=8$, $D_{\text{head}}=128$ ($128\text{ KiB/token}$).
  - **Ministral 3 3B Base:** $L=26$, $H_{KV}=8$, $D_{\text{head}}=128$ ($104\text{ KiB/token}$).
  - **SmolLM3 3B:** $L=36$, $H_{KV}=4$, $D_{\text{head}}=128$ ($72\text{ KiB/token}$).
  - **Qwen 3 1.7B:** $L=28$, $H_{KV}=8$, $D_{\text{head}}=128$ ($112\text{ KiB/token}$).
  Because $d=128$ is a power of 2 ($2^7$), the Walsh-Hadamard Transform executes via a fast radix-2 butterfly with zero padding required.

### 5.2 Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap)

- **VRAM Footprint Reduction:**
  In native BF16, storing 32K context for Qwen 3 4B requires $4.50\text{ GB}$ of VRAM. Combined with $8.06\text{ GB}$ of static model weights, this consumes $12.56\text{ GB}$ VRAM, exceeding an 8 GB RTX 4060 and leaving only $3.44\text{ GB}$ headroom on the 16 GB HPC cap (preventing multi-user concurrency).
  Under 4-bit asymmetric quantization (`KQV`):
  - Keys stored at 4-bit scalar ($16\text{ bytes/token/head}$ vs. $64\text{ bytes}$).
  - Values stored at 3-bit scalar + 1-bit QJL ($16\text{ bytes/token/head}$ vs. $64\text{ bytes}$).
  - Total KV cache size shrinks by **$4\times$**, reducing 32K context to **$1.125\text{ GB}$**.
  - **Fits easily on local 8 GB RTX 4060:** $8.06\text{ GB}$ weights (or Qwen 3 1.7B at $3.5\text{ GB}$) + $1.125\text{ GB}$ cache operates comfortably within hardware limits.
  - **Enables multi-user SME scaling on 16 GB cap:** 10 concurrent sessions at 4K context require only $1.41\text{ GB}$ of VRAM instead of $5.63\text{ GB}$.

### 5.3 Concrete Integration Path in `kv_bench`

1. **`WP-3` Micro-Benchmarking (Quantization Quality & Statistical Profiling):**
   - Implement the **Shannon-$\phi$ plane** decomposition ($[\epsilon^{\text{snr}}_K, \epsilon^{\text{dir}}_K]$, $[\epsilon^{\text{snr}}_V, \epsilon^{\text{dir}}_V]$, $[\epsilon^{\text{snr}}_T, \epsilon^{\text{dir}}_T]$) into our microbenchmark evaluation suite.
   - Benchmark synthetic vs. real calibration activations extracted from `Qwen 3 4B` and `Phi-4 mini` to verify whether real LLM key caches suffer from the low-rank pathology identified in Section 8.
   - Measure empirical Mann-Whitney $r$ and Energy Distance across bit budgets $n \in \{2, 3, 4, 8\}$ to validate the $n=4$ sweet spot on our pinned models.

2. **`WP-6` Engine Integration (`vLLM` PagedAttention & `LMCache`):**
   - **Kernel Implementation:** Implement the Fast Walsh-Hadamard Transform (FWHT) as a fused Triton or CUDA kernel operating on $128$-element vectors within the shared memory of PagedAttention blocks.
   - **Asymmetric Paged KV Storage:** Configure `vLLM` / `LMCache` block allocators to store Key pages in 4-bit Lloyd-Max packed format, while Value pages store packed 3-bit codes plus 1-bit binary sign words.
   - **Avoid QJL on Keys:** Strictly avoid any 1-bit randomized sketching on Key vectors, adhering to the mathematical proof of the $2\pi$ variance penalty.
   - **Diagnostic Warnings:** Implement monitoring for worst-case attention logit KL divergence and top-token argmax stability during long-context generation rather than relying on average token perplexity.

- **Classification for Project:** **Core Foundational Reference for Structural Quantization Architecture** (Directly dictates our asymmetric K-V precision strategy and provides our mathematical evaluation metrics for WP-3 and WP-6).
