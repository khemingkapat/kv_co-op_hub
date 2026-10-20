# LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory

## 0. Plain-English TL;DR (What It Tests & Why We Use It)

- **What This Benchmark Is:**
  LongMemEval is a rigorous conversational memory benchmark designed to test whether an AI assistant can remember, update, and reason about personal user facts across dozens of chat sessions spanning days, weeks, or months of simulated real-world interaction.

- **Why Standard Tests Aren't Enough:**
  - Simple synthetic benchmarks like standard **Needle In A Haystack (NIAH)** only test retrieving a single, artificially inserted sentence from a static document.
  - Basic **Perplexity ($\Delta\text{PPL}$)** checks only measure general language predictability, completely missing whether an eviction policy accidentally threw away a critical personal fact (like an allergy or an updated schedule).
  - LongMemEval mimics real life: evidence is scattered across up to 6 separate chats, facts are stated casually and indirectly (e.g., asking about car insurance rather than bluntly stating "I bought a car"), and information evolves over time.

- **How It Works Step-by-Step:**
  1. *Curated Questions & Facts:* 500 expert-designed questions test five core abilities (extracting details, multi-session aggregation, tracking changed facts, reasoning over dates/timestamps, and correctly saying "I don't know" when information was never mentioned).
  2. *Realistic Conversation Haystack:* Evidence statements are woven into natural multi-turn chats and padded with hundreds of realistic, timestamped conversational sessions drawn from real-world datasets (ShareGPT, UltraChat).
  3. *Flexible Length Scaling:* Histories can be stacked to test models at custom lengths (standard benchmarks provide $\approx 115\text{K}$ tokens in $\text{LongMemEval}_S$ and up to $1.5\text{M}$ tokens in $\text{LongMemEval}_M$).

---

## 1. Key Takeaways & How We Use It in `kv_bench`

- **Specific KV-Cache Failures Exposed:**
  - **Query-Dependent Eviction Breakdown:** Techniques that evict tokens based solely on the current prompt (like naive SnapKV) fail on multi-session questions because evidence needed in the current turn was already pruned turns ago.
  - **Information Overwrite:** Memory systems that aggressively compress past chats often overwrite fine details, causing the model to hallucinate or misattribute facts across different sessions.
  - **Timestamp Blindness:** If a compression codec strips or blurs positional and temporal tokens, the model cannot distinguish between older and newer facts during knowledge updates.

- **Key Findings from the Paper Worth Knowing:**
  1. **Long-Context Models Still Suffer Severe Memory Drops:** Even state-of-the-art long-context LLMs (GPT-4o, LLaMA-3.1-70B/8B, Phi-3) drop $30\%\text{--}60\%$ in accuracy when reading long multi-session histories compared to reading only the evidence sessions.
  2. **Round-Level Granularity Beats Full-Session Granularity:** Slicing conversations at the turn/round level ($1\text{ user message} + 1\text{ assistant reply}$) retains far more relevant detail for retrieval and attention than trying to compress entire multi-topic sessions as single blocks.
  3. **Multi-Session Reasoning is the Hardest Task:** Models struggle most when they have to find and combine multiple separate clues across different conversations (e.g., tallying items mentioned on different days).

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 23 in `manual_reviews.md` provided an excellent breakdown of the 4 key differences from standard NIAH (multiple needles/sessions, indirect facts, realistic timestamped haystacks, and 5 memory skills).
  - *Refinement from Document:* While the benchmark evaluates retrieval-augmented memory systems, its scalable session-stacking pipeline can be directly applied to evaluate KV-cache compression and tiering frameworks (as demonstrated in the EpiCache paper, which evaluated directly on LongMemEval).

---

## 2. Workload & Diagnostic Profile

- **Publication & Authors:** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu (UCLA, Tencent AI Lab Seattle, UC San Diego). Published at ICLR 2025. (arXiv:2410.10813v2)
- **Target Workload & Context Lengths:**
  - Standard evaluation suites: $\text{LongMemEval}_S$ ($\approx 115\text{K}$ tokens, $\approx 50$ sessions per problem) and $\text{LongMemEval}_M$ (500 sessions, $\approx 1.5\text{M}$ tokens).
  - Configurable context sweeps: Evaluated under controlled context lengths of $20\text{K}, 40\text{K}, 60\text{K}, 80\text{K},$ and $100\text{K}$ tokens.

- **The Five Core Memory Capabilities Tested:**
  1. *Information Extraction (IE):* Recalling specific single-session facts stated by the user or assistant.
  2. *Multi-Session Reasoning (MR):* Synthesizing facts across multiple disparate sessions (e.g., calculating total hours worked across seasons).
  3. *Knowledge Updates (KU):* Identifying when a user's life state has changed (e.g., changing travel plans from Hawaii to Paris) and answering with the updated state.
  4. *Temporal Reasoning (TR):* Calculating elapsed time or chronological sequence using timestamps and colloquial time references.
  5. *Abstention (ABS):* Identifying unmentioned or false-premise questions and correctly refusing to answer rather than hallucinating.

- **Evaluation Methodology:**
  - Model responses are evaluated using an expert-prompted GPT-4o judge against human ground-truth answers (achieving $>97\%$ agreement with human experts).

---

## 3. Role in Senior Project ([`docs/proposal.md`](../../proposal.md) Section 3.3.2)

- **Role:** **Final Report Metric — Task Quality Check**
  - Used in `WP-7` to verify that our 3-axis co-optimized KV cache maintains long-term memory fidelity under our simulated 10–15 user multi-turn workload.
  - Prevents the team from relying solely on perplexity or synthetic passkey retrieval, ensuring our compression policies don't break genuine interactive dialogue capabilities.
- **Execution Strategy on 8–16 GB GPUs:**
  - Rather than running the full 1.5M token suite, we utilize LongMemEval's session stacking to generate standardized evaluation sets scaled to our hardware context targets ($4\text{K}\text{--}32\text{K}$ tokens).
  - We run the uncompressed baseline (native vLLM FP16/BF16) on the same subset first to calibrate the performance ceiling before evaluating our co-optimized configurations.
