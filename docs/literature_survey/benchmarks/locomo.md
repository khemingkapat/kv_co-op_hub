# LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents

## 0. Plain-English TL;DR (What It Tests & Why We Use It)

- **What This Benchmark Is:**
  LoCoMo is a comprehensive conversational memory benchmark designed to evaluate how well large language model agents remember, track, summarize, and respond across very long human-like relationships spanning weeks or months, up to thirty-five separate chat sessions and over nine thousand words of conversation.

- **Why Standard Tests Aren't Enough:**
  - Simple synthetic tests like Needle In A Haystack only check whether a model can spot a single artificial sentence planted inside an irrelevant wall of text.
  - Perplexity checks only verify that language remains grammatically smooth, completely failing to notice if a memory system dropped a critical personal event or mixed up which person said what.
  - Existing multi-session dialogue benchmarks are too short, rarely exceeding five chat sessions or one thousand tokens.
  - LoCoMo tests the real messiness of long-term human dialogue: personal timelines where events evolve chronologically, conversations grounded in evolving life graphs, image sharing, and multi-session questions that require connecting clues across weeks of chats.

- **How It Works Step-by-Step:**
  1. *Grounded Agent Generation:* Two artificial agents are assigned detailed personalities and a chronological graph of life events covering six to twelve months. They chat naturally over up to thirty-five sessions, referencing past happenings, sharing photos with captions, and reflecting on earlier conversations.
  2. *Human Verification & Cleanup:* Human annotators manually review the generated chats, removing inconsistencies, correcting broken image references, and making sure dialogue strictly adheres to the underlying life-event timeline (editing roughly fifteen percent of all turns).
  3. *Three Realistic Evaluation Tasks:*
     - *Question Answering:* Models answer single-session, multi-session, time-tracking, open-knowledge, and trick/adversarial questions about the conversation history.
     - *Event Summarization:* Models must read the entire conversation and extract the underlying timeline of major life events, scored on factual accuracy against the ground-truth event graph.
     - *Dialogue Generation:* Models continue the conversation and react to shared images while trying to maintain long-term narrative consistency with their established persona.

---

## 1. Key Takeaways & How We Use It in `kv_bench`

- **Specific KV-Cache Failures Exposed:**
  - **Speaker Misattribution:** Aggressive token pruning or uncalibrated quantization frequently blurs speaker identities, causing the model to confuse which speaker went to college, who owns a pet, or who got injured.
  - **Temporal Sequence Scrambling:** If an eviction or compression policy drops time cues, timestamps, or date references, the model cannot answer questions about how much time has passed or whether an event happened before or after another.
  - **The Long-Context Hallucination Trap (Adversarial Collapse):** As the KV cache fills up with thousands of past tokens, models become highly vulnerable to trick questions. Instead of recognizing that an event never happened, models with bloated or noisy context hallucinate fake confirmations.
  - **Multi-Session Disconnection:** Policies that evict tokens based solely on the immediate prompt (like naive attention-sink or local-window pruning) permanently discard details from early sessions that are required for multi-hop questions later on.

- **Key Findings from the Paper Worth Knowing:**
  1. **Long Context Windows Suffer Catastrophic Adversarial Collapse:** While giving a model a larger context window improves recall on straightforward questions, its ability to resist trick or unanswerable questions completely collapses. When fed sixteen thousand tokens of history, commercial models dropped from seventy percent down to two percent accuracy on adversarial questions, falsely confirming fabricated events because the sheer volume of context distracted the attention mechanism.
  2. **Temporal Reasoning is the Hardest Task for Language Models:** Even the strongest commercial models lag behind humans by more than seventy percent on time-based questions (such as calculating elapsed days or ordering separated events).
  3. **Structured Atomic Summaries Beat Raw Token Dumping:** When using retrieval to answer questions, converting past dialogues into concise factual assertions (called observations) achieved higher accuracy than retrieving raw dialogue transcripts or high-level session summaries. Furthermore, for understanding entire life timelines, step-by-step incremental summarization beat dumping the entire raw dialogue into a giant context window.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 25 in `manual_reviews.md` correctly identified the mixed-modal nature, the chronological timeline and event-graph grounding, the three evaluation tasks (question answering, event summarization, and dialogue generation), and human-in-the-loop cleaning.
  - *Confirmed:* The team's brainstorming intuition that incremental indexing outperforms dumping the entire history into a single prompt is directly proven by the paper's event summarization experiments.
  - *Refinements from the PDF Ground Truth:* The team notes did not capture the catastrophic collapse on adversarial questions when context scales up, nor the dramatic seventy-three percent gap in temporal reasoning. Additionally, while the authors evaluated retrieval-augmented systems and full-context models, newer KV-cache tiering architectures (most notably Apple's EpiCache in ICML 2026) benchmark directly on LoCoMo to evaluate episodic cache compression and offloading over multi-session dialogue.

---

## 2. Workload & Diagnostic Profile

- **Publication & Authors:**
  - Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang (University of North Carolina at Chapel Hill, University of Southern California, Snap Inc.).
  - Published at ACL / arXiv:2402.17753v1 (February 2024).
  - Code and dataset: `https://snap-research.github.io/locomo`.

- **Dataset Scale & Target Workloads:**
  - **Total Conversations:** 50 multi-session conversations.
  - **Sessions per Conversation:** Average $19.3$ sessions (ranging up to $35$ sessions).
  - **Turns per Conversation:** Average $304.9$ dialogue turns ($15.8$ turns per session).
  - **Context Length:** Average $9,209.2$ tokens per conversation (dialogue turns average $30.2$ tokens; session summaries average $127.4$ tokens; atomic observations average $18.2$ tokens).
  - **Comparison with Prior Work:** LoCoMo is $9\times$ longer in tokens, has $6\times$ more turns, and covers $4\times$ more sessions than Multi-Session Chat (MSC).
  - **Multimodal Scale:** Average $32.3$ shared images per conversation.

- **The Three Core Benchmark Tasks:**

  ### Task 1: Question Answering (7,512 Evaluated Questions)
  Evaluates factual retrieval and multi-session reasoning across 5 distinct categories:
  1. *Single-Hop Retrieval ($2,705$ questions, $36.0\%$):* Requires extracting a fact contained within a single conversational session.
  2. *Multi-Hop Retrieval ($1,104$ questions, $14.6\%$):* Requires synthesizing and cross-referencing facts scattered across multiple distinct sessions.
  3. *Temporal Reasoning ($1,547$ questions, $20.6\%$):* Requires reasoning about timestamps, relative dates, and the chronological ordering of life events.
  4. *Open-Domain Knowledge ($285$ questions, $3.9\%$):* Requires connecting dialogue facts with external world knowledge and common sense.
  5. *Adversarial / Abstention ($1,871$ questions, $24.9\%$):* Trick questions containing false premises or unmentioned events, testing if the model correctly abstains rather than hallucinating.

  *Evaluation Metric:* Token-level partial match $F_1$ score normalized against human ground-truth answers directly extracted from transcripts, plus $\text{Recall@}k$ for retrieved context.

  ### Task 2: Event Summarization (Chronological Event Graphs)
  Evaluates whether the model can extract the underlying sequence of major life events across the entire conversation history:
  - Ground truth: Causal event graph $G = (V, E)$ where nodes $e_i \in V$ represent life events with occurrence dates $t_i$, and directed edges $l = (e_i, e_j) \in E$ represent causal dependencies.
  - Average $24.2$ ground-truth events per conversation ($896.5$ tokens).
  - *Evaluation Metrics:* Standard ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) and FactScore decomposing generated text into atomic facts to calculate:
    $$\text{Precision} = \frac{|\text{Atomic Facts in Summary} \cap \text{Atomic Facts in } G|}{|\text{Atomic Facts in Summary}|}$$
    $$\text{Recall} = \frac{|\text{Atomic Facts in Summary} \cap \text{Atomic Facts in } G|}{|\text{Atomic Facts in } G|}$$
    $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
    *Plain English:* Precision measures what fraction of the model's generated statements actually happened in the person's life; Recall measures what fraction of the person's true life events the model successfully captured; $F_1$ balances both.

  ### Task 3: Multimodal Dialogue Generation
  Evaluates generating the next conversational turn conditioned on history:
  - Tested on MiniGPT-5 fine-tuned on MMDialog.
  - *Evaluation Metrics:* BLEU-1/2, ROUGE-L, and MM-Relevance (semantic alignment between generated dialogue and shared images).

- **Empirical Baseline Numbers from the Paper:**

  ### Table 1: Question Answering Performance ($F_1$ Score)
  | Model | Context Window | Single-Hop | Multi-Hop | Temporal | Open-Domain | Adversarial | Overall $F_1$ |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Human Baseline** | — | **$95.1$** | **$85.8$** | **$92.6$** | **$75.4$** | **$89.4$** | **$87.9$** |
  | **Mistral-Instruct-7B** | 8K | $10.2$ | $12.8$ | $16.1$ | $19.5$ | $17.0$ | $13.9$ |
  | **LLaMA-2-Chat-70B** | 4K | $19.7$ | $14.4$ | $13.3$ | $15.9$ | $22.1$ | $17.9$ |
  | **GPT-3.5-turbo** | 4K | $29.9$ | $23.3$ | $17.5$ | $29.5$ | $12.8$ | $22.4$ |
  | **GPT-4-turbo** | 4K | $23.4$ | $23.4$ | $10.4$ | $24.6$ | **$70.2$** | $32.1$ |
  | **GPT-3.5-turbo-16K** | 4K | $31.7$ | $25.4$ | $16.8$ | $27.6$ | $13.1$ | $24.1$ |
  | **GPT-3.5-turbo-16K** | 8K | $38.8$ | $31.2$ | $21.0$ | $35.0$ | $8.4$ | $25.2$ |
  | **GPT-3.5-turbo-16K** | 12K | $51.1$ | $40.4$ | $25.0$ | $36.5$ | $6.4$ | $33.5$ |
  | **GPT-3.5-turbo-16K** | 16K (Full Context) | **$56.4$** | **$42.0$** | $20.3$ | **$37.2$** | $2.1$ | **$37.8$** |

  ### Table 2: RAG vs. Memory Representation (Reader: GPT-3.5-turbo-16K, Retriever: DRAGON)
  | Context Representation | Top-$k$ Units | Single-Hop $F_1$ | Multi-Hop $F_1$ | Temporal $F_1$ | Adversarial $F_1$ | Overall $F_1$ | Retrieval Recall ($\text{R@}k$) |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **No Retrieval (4K Raw Truncated)** | — | $29.9$ | $23.3$ | $17.5$ | $12.8$ | $22.4$ | — |
  | **Raw Dialogue Turns** | $k=5$ | $42.9$ | $19.4$ | $21.3$ | $31.9$ | $31.7$ | $58.8\%$ |
  | **Raw Dialogue Turns** | $k=25$ | $48.1$ | $36.1$ | $26.2$ | $23.4$ | $35.8$ | $79.9\%$ |
  | **Raw Dialogue Turns** | $k=50$ | $50.9$ | $37.2$ | $24.6$ | $17.0$ | $34.8$ | $84.8\%$ |
  | **Atomic Observations** | $k=5$ | $44.3$ | $30.6$ | **$41.9$** | **$44.7$** | **$41.4$** | $49.6\%$ |
  | **Atomic Observations** | $k=25$ | $44.6$ | $33.2$ | $41.8$ | $27.7$ | $38.0$ | $66.0\%$ |
  | **Atomic Observations** | $k=50$ | $44.0$ | $34.5$ | $41.1$ | $27.7$ | $37.8$ | $71.1\%$ |
  | **Session Summaries** | $k=2$ | $34.6$ | $15.7$ | $26.9$ | $36.2$ | $29.9$ | $61.5\%$ |
  | **Session Summaries** | $k=10$ | $34.5$ | $14.7$ | $29.3$ | $40.4$ | $31.5$ | **$90.7\%$** |

  ### Table 3: Event Summarization Performance (FactScore & ROUGE)
  | Model | Context Mode | ROUGE-1 | ROUGE-2 | ROUGE-L | FactScore Precision | FactScore Recall | FactScore $F_1$ |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Mistral-Instruct-7B** | 8K Incremental | $29.4$ | $7.2$ | $14.1$ | $27.1\%$ | $19.8\%$ | $23.0$ |
  | **LLaMA-2-Chat-70B** | 4K Incremental | $28.1$ | $9.3$ | $14.8$ | $36.3\%$ | $22.7\%$ | $28.3$ |
  | **GPT-4-turbo** | 4K Incremental | $38.8$ | $11.4$ | $20.6$ | **$51.6\%$** | $41.8\%$ | $45.1$ |
  | **GPT-3.5-turbo** | 4K Incremental | **$41.1$** | **$13.5$** | **$20.9$** | $45.3\%$ | **$46.5\%$** | **$45.9$** |
  | **GPT-3.5-turbo-16K** | 16K Monolithic Window | $36.2$ | $8.5$ | $16.4$ | $42.3\%$ | $37.8\%$ | $39.9$ |

  > [!IMPORTANT]
  > **Key Architectural Takeaway from Table 3:** Feeding the entire conversation into a wide 16K context window resulted in **lower** summarization quality ($39.9\text{ FactScore } F_1$) than feeding the conversation incrementally session-by-session into a standard 4K window ($45.9\text{ FactScore } F_1$). Uncompressed token accumulation introduces attention dilution and noise, degrading factual recall by $8.7\%$.

- **Five Dominant Error Modes in Long-Term Conversational Memory:**
  1. *Missing Information:* Omission of critical life events due to failure in resolving causal or temporal chains across distant sessions.
  2. *Hallucination & Cross-Event Padding:* Blending details from unrelated events into a single false statement (e.g., claiming someone's vegan ice cream was praised at a gaming tournament).
  3. *Misunderstanding Conversational Nuances:* Interpreting sarcasm, teasing, or humorous hypotheticals as factual historical life events.
  4. *Speaker Misattribution:* Assigning actions, hobbies, or injuries to the wrong conversational participant.
  5. *Saliency Misjudgments:* Elevating casual pleasantries ("How have you been?") into major life milestones while missing genuine life milestones.

---

## 3. Role in Senior Project ([`docs/proposal.md`](../../proposal.md) Section 3.3.2)

- **Role:** **Task Quality Check / Multi-Turn Conversational Memory Benchmark (`WP-7`)**
  - Serves as the primary real-world conversational memory validation suite alongside LongMemEval.
  - Evaluates whether our 3-axis co-optimized KV cache (priority scoring + selective INT4/INT8 quantization + CPU offloading) can preserve multi-session coherence without triggering catastrophic forgetting or speaker misattribution.
  - Prevents the team from relying solely on synthetic Needle In A Haystack (NIAH) or static document perplexity ($\Delta\text{PPL}$), which fail to catch conversational amnesia, temporal inversion, or hallucination.

- **Why LoCoMo is an Ideal Hardware Fit for Our Consumer GPUs:**
  - **Context Length Alignment:** LoCoMo's average conversation length is $9,209.2$ tokens ($\approx 9\text{K}\text{--}12\text{K}$ tokens).
  - Unlike LongMemEval$_M$ ($1.5\text{M}$ tokens) or synthetic $128\text{K}$ tests that require massive synthetic downsampling or multi-GPU clusters, LoCoMo conversations fit natively within the $16\text{K}\text{--}32\text{K}$ native context window of our pinned models (`Qwen 3 4B`, `Phi-4 mini`, `SmolLM3 3B`, `Ministral 3 3B`, `Qwen 3 1.7B`).
  - On our local 8 GB RTX 4060 and Slurm 16 GB RTX 4090 cap, a $9\text{K}$-token conversation in native BF16 consumes:
    $$\text{KV VRAM (BF16)} = 9,209 \times \text{Bytes/Token}$$
    - For `Qwen 3 4B` ($144\text{ KiB/token}$): $1.26\text{ GB}$ of KV cache.
    - For `Phi-4 mini` ($128\text{ KiB/token}$): $1.12\text{ GB}$ of KV cache.
    - For `Qwen 3 1.7B` ($112\text{ KiB/token}$): $0.98\text{ GB}$ of KV cache.
  - Under concurrent multi-session serving (10–15 simulated users), total KV cache reaches $12.6\text{--}18.9\text{ GB}$, instantly overflowing the 16 GB VRAM budget (which only has $\approx 7.94\text{ GB}$ of headroom after model weights).
  - This allows `kv_bench` to stress-test our tiered memory hierarchy (keeping active turn KV tokens in GPU VRAM while quantizing and offloading past chat sessions to host RAM over PCIe Gen 4) under a realistic dialogue workload.

- **Execution Strategy in `kv_bench`:**
  1. *Text + Caption Evaluation:* In accordance with the paper's experimental protocol (§5) and our `--language-model-only` requirement (`docs/MODELS.md`), images are replaced with their verified BLIP-2 captions. This eliminates the need to run vision backbones or allocate VRAM for multimodal encoders ($0.8\text{--}1.5\text{ GB}$ saved).
  2. *Baseline Calibration:* Run the uncompressed full-precision baseline (native vLLM BF16) on the 50 conversations to establish the exact accuracy upper bound for our pinned models.
  3. *Co-Optimization Sweep:* Evaluate our 3-axis engine configurations against the baselines:
     - Compare Question Answering $F_1$ across all 5 reasoning types, with special diagnostic attention to *Temporal Reasoning* and *Adversarial Abstention* (to ensure our compression does not trigger the long-context hallucination trap).
     - Measure Event Summarization FactScore to verify that hierarchical cache compression retains causal event structures across distant sessions.
     - Record system metrics: Peak VRAM reduction, Time-To-First-Token (TTFT), decoding throughput, host RAM delta, and PCIe transmission bandwidth.
