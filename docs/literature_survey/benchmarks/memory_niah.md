# Memory NIAH: Needle in the Haystack for Memory Based Large Language Models

## 0. Plain-English TL;DR (What It Tests & Why We Use It)

- **What This Benchmark Is:**
  Memory NIAH is an investigative benchmark designed to test whether artificial intelligence models that use compressed or external memory systems can actually retrieve isolated facts hidden deep inside long documents. It puts models through passkey and needle-in-the-haystack pressure tests across context lengths from twenty-four thousand up to one million words and tokens.

- **Why Standard Tests Aren't Enough:**
  - Simple perplexity checks only test whether a model can predict normal English flow, completely hiding whether an eviction or compression policy threw away a critical fact.
  - When memory-saving designs (such as sliding-window attention, recurrent memory states, or prompt compression) try to shrink long documents into small internal states, they often blur or overwrite older facts.
  - Standard needle-in-the-haystack tests usually only test huge full-attention models. Memory NIAH specifically compares how different memory architectures behave when searching for facts hidden across different positions (the beginning, middle, and end) of massive contexts.

- **How It Works Step-by-Step:**
  1. *Hiding the Needle:* A simple key sentence (such as a random multi-digit passkey or a specific factual statement like a favorite sandwich spot) is inserted into a large background document at controlled depths (ranging from the start to the middle and end).
  2. *Background Haystack:* The test embeds the needle into repetitive distractor text or dozens of real-world essays (such as Paul Graham essays spanning up to one hundred and thirty-seven thousand tokens).
  3. *Retrieval Query & Evaluation:* The document ends with an open-ended retrieval prompt asking for the secret value. The model is scored on whether it reproduces the exact key digits or passes a sequence-matching check on the requested target sentence.

---

## 1. Key Takeaways & How We Use It in `kv_bench`

- **Specific KV-Cache Failures Exposed:**
  - **Recency Bias & Severe Early Blind Spots:** Recurrent memory and sliding-window architectures suffer catastrophic recall failure on facts placed at the beginning or middle of the sequence. While they achieve near-perfect retrieval for needles near the very end, recall drops to single digits for earlier context.
  - **Information Blurring from Aggressive Compression:** Compacting long context into a fixed-size representation mixes together different segments, making fine-grained numbers impossible to distinguish during generation.
  - **Reconstruction Bottlenecks:** Even when a memory system locates the correct text segment, generating multi-digit numbers or longer statements requires sufficient representational capacity; small latent vectors lose detail when generating longer answers.

- **Key Findings from the Paper Worth Knowing:**
  1. **Zero-Shot Recurrent Memory Collapses on Early Context:** Without expensive task-specific fine-tuning, models that compress history into recurrent state matrices (like Infini-Transformer) achieve only six to fourteen percent accuracy when retrieving facts from the beginning or middle of a long document, despite achieving ninety-nine to one hundred percent on the final segment.
  2. **Standard Long-Context Models Deteriorate on Real Documents:** Open-source long-context transformers struggle significantly as context grows in real essays; at one hundred thousand tokens, a three-point-eight billion parameter model retrieved the needle only twenty-six to thirty-seven percent of the time.
  3. **Prefix Keys Enable Position-Invariant Retrieval:** Structuring associative memory lookups with a short fixed prefix (matching the prompt format) allows the system to look up stored text chunks regardless of where they were placed in the document.
  4. **Offloading Memory to Host RAM Eliminates GPU Scaling Overhead:** By keeping the large associative memory table entirely on system host RAM and only sending the single retrieved embedding back to the GPU for token decoding, the GPU memory footprint remains completely flat as context scales from short sequences to one million tokens.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 29 in `manual_reviews.md` accurately identified that specialized memory and sliding-window architectures frequently exhibit severe retrieval blind spots on early or middle tokens, and correctly designated this benchmark as a controlled diagnostic test to see if eviction or offloading policies drop information.
  - *Refinement from Document Ground Truth:* The paper specifically evaluates IBM's associative memory architecture (Larimar) against sliding window, Infini-Transformer, and dense transformers (Mistral 7B, Phi-3-mini). The empirical data reveals that zero-shot compressive memory suffers an extreme start-versus-end disparity (dropping to six to eleven percent at the start while retaining near one hundred percent at the end). This provides mathematical justification for why our project avoids destructive lossy recurrent compression in favor of our three-axis design (preserving exact KV tokens via selective quantization and PCIe CPU offloading).

---

## 2. Workload & Diagnostic Profile

- **Publication & Authors:** Elliot Nelson, Georgios Kollias, Payel Das, Subhajit Chaudhury, Soham Dan (IBM Research). Published on arXiv (arXiv:2407.01437v2 [cs.CL], July 2024).
- **Target Workload & Context Lengths:**
  - *Passkey Context Lengths:* Evaluated at $128\text{K}$ ($131,072$ tokens), $1\text{M}$ ($1,048,576$ tokens), and extended to $1.2\text{M}$ ($1,200,057$ tokens).
  - *Needle-in-the-Haystack (NIAH) Context Lengths:* Evaluated on the Paul Graham essay haystack at $\approx 137\text{K}$ tokens (full haystack), $100\text{K}$ tokens ($5,000$ sentences for Phi-3-mini), and $24\text{K}$ tokens ($1,200$ sentences for Mistral 7B v0.2).
  - *Needle Variations:* Multi-digit passkeys (3 to 8 random digits: *"The pass key is 9054"*), magic numbers (*"The magic number is X"*), and semantic sentence needles (*"The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."*).

- **Associative Memory Mathematical Formulation (Larimar Reference):**
  To avoid quadratic self-attention while preventing positional information loss, the paper partitions context into $N$ segments and maps them into an external associative memory matrix:
  
  $$\min_M \|Z - WM\|_2^2 \implies M = W^\dagger Z$$
  
  - $M \in \mathbb{R}^{K \times C}$: The stored associative memory matrix, where $K$ is the number of memory slots and $C$ is the latent embedding dimension.
  - $Z \in \mathbb{R}^{N \times C}$: The matrix of segment latent encodings generated by the neural encoder for all $N$ context segments.
  - $W \in \mathbb{R}^{N \times K}$: The matrix of key vectors corresponding to each segment.
  - $W^\dagger$: The Moore-Penrose pseudoinverse of $W$. When key vectors are orthogonal one-hot assignments, $W^\dagger = W^T$, simplifying computation from $O(K^3)$ to $O(N)$ runtime.

  During the read/retrieval phase, a query key vector $w$ retrieves a conditioned latent vector for the decoder:
  
  $$z_{\text{read}} = w M$$
  
  - $z_{\text{read}} \in \mathbb{R}^C$: The single reconstructed latent vector fed into the generative decoder to condition output generation.
  - $w \in \mathbb{R}^K$: The query key vector computed from the incoming user question.
  - $M \in \mathbb{R}^{K \times C}$: The associative memory matrix stored on the CPU host.

  To guarantee reliable addressing between a prompt query and stored facts, keys are computed from fixed-length prefix encodings $\tilde{z}$ (e.g., the first 4 words of a sentence) using nearest-neighbor lookup:
  
  $$k^*(\tilde{z}) := \arg\min_k \|\tilde{z} - \tilde{M}_{k, :}\|_2, \quad w_k = \mathbf{1}(k = k^*(\tilde{z}))$$
  
  - $k^*(\tilde{z})$: The row index in key memory matrix $\tilde{M}$ whose prefix embedding has the smallest Euclidean distance to query prefix encoding $\tilde{z}$.
  - $\tilde{M}_{k, :}$: The $k$-th row of the fixed key memory matrix $\tilde{M} \in \mathbb{R}^{K \times C}$.
  - $w_k$: The $k$-th element of the resulting one-hot key vector $w$.
  - $\mathbf{1}(\cdot)$: The indicator function returning 1 if the condition is met and 0 otherwise.

- **Empirical Findings & Benchmark Tables:**

  *Table 1: Passkey Retrieval Success (%) across Needle Positions (Start / Middle / End):*
  | Model | Context Length: 128K | Context Length: 1M |
  | :--- | :---: | :---: |
  | **Larimar (1.3B)** | **100% / 100% / 100%** | **100% / 100% / 100%** |
  | **Infini-Transformer (Linear)** *(Zero-Shot)* | 11% / 14% / 100% | 8% / 6% / 98% |
  | **Infini-Transformer (Linear + $\delta$)** *(Zero-Shot)* | 6% / 9% / 99% | 7% / 6% / 97% |

  *Table 2: Larimar 1.3B Retrieval Accuracy across Passkey Lengths at 1.2M Tokens:*
  | Metric / Digit Count | 3 Digits | 4 Digits | 5 Digits | 6 Digits | 7 Digits | 8 Digits |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Recall (%) on 100 Random Passkeys** | **100%** | **86%** | **87%** | **79%** | **59%** | **50%** |

  *Table 3: Needle-in-the-Haystack Performance on Paul Graham Essays (~137K Haystack):*
  | Model | Evaluated Context | 3-Digit Magic Number | 4-Digit Magic Number | San Francisco Needle (ROUGE-L) |
  | :--- | :---: | :---: | :---: | :---: |
  | **Larimar (1.3B, Prefix Keys)** | **137K** | **95%** | **64%** | **1.00** |
  | **Larimar (1.3B, No Prefix)** | 137K | 88% | 14% | 0.00 |
  | **Mistral 7B v0.2** | 24K *(1,200 sentences)* | 66% | 62% | 0.80 |
  | **Phi-3-mini-128K (3.8B)** | 100K *(5,000 sentences)* | 27% | 26% | 0.37 |

- **Latency, Bandwidth & Systems Footprint:**
  - *Tail Latency (P95/P99):* **Not reported in paper** (mean retrieval accuracy only).
  - *PCIe Transfer Volume:* **Not reported in paper**.
  - *Hardware Footprint Analysis:* All associative memory writes and reads execute on CPU host memory, while GPU execution is strictly limited to sequence segment encoding and decoding conditioned on a single latent vector ($z_{\text{read}} \in \mathbb{R}^C$, decoder sequence length $\le O(100)$ tokens). This keeps GPU VRAM usage completely decoupled from context length scaling.

---

## 3. Role in Senior Project ([`docs/proposal.md`](../../proposal.md) Section 3.3.2)

- **Role:** **Diagnostic Metric — Positional Retrieval & Eviction Blind-Spot Falsification**
  - In [`docs/proposal.md`](../../proposal.md) Section 3.3.4, **Positional Retrieval** is defined as an isolated diagnostic metric triggered when a primary quality metric (such as Long-Term Memory Adherence on LongMemEval or Logit Divergence) flags degradation.
  - Used in `WP-3` (micro-benchmarking eviction and pruning heuristics) and `WP-6` (evaluating co-optimized serving under vLLM and LMCache) to verify whether our 3-axis policies create spatial "blind spots" (e.g., dropping early dialogue context or intermediate conversation turns).
  - Contrasts directly with generic perplexity ($\Delta\text{PPL}$): as demonstrated by the Infini-Transformer collapse in Table 1, a system can exhibit low training perplexity while failing to retrieve facts located at $0\%\text{--}50\%$ context depth.

- **Execution Strategy on 8–16 GB GPUs (`docs/MODELS.md`):**
  - **Context Scaling Windows:** Evaluated across context sweeps from $4\text{K}$ to $32\text{K}$ tokens, matching the dedicated VRAM headroom on our Slurm RTX 4090 (16 GB cap) and local RTX 4060 (8 GB).
  - **Depth Sweeps:** Needles are distributed uniformly across 10 depth deciles ($0\%, 10\%, \dots, 90\%, 100\%$) across the context sequence to produce a full 2D retrieval heat map (Context Length $\times$ Document Depth).
  - **Target Models Evaluated:**
    - *Dense GQA Baselines:* `Qwen 3 4B Instruct`, `Phi-4 mini Instruct`, `Ministral 3 3B Base`, `SmolLM3 3B`, and `Qwen 3 1.7B` (dense attention workhorses).
    - *Hybrid / SWA Baselines:* `Gemma 4 E4B Instruct` (sliding window / hybrid attention).
  - **Falsification Threshold:** An eviction or tiering policy is flagged as defective if retrieval accuracy drops below $95\%$ at any depth decile within the target $32\text{K}$ context window compared to the uncompressed FP16 baseline.
