# Literature Review Specification & Execution Playbook (Temporary / One-Off)

> **Context:** Senior Project — Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs  
> **Target Repository:** `kv_co-op_hub`  
> **Agent Skill:** [`.agents/skills/literature-review/SKILL.md`](.agents/skills/literature-review/SKILL.md)  
> **Team Skimming Notes (Verify Against PDF):** [`manual_reviews.md`](manual_reviews.md)  
> **Local PDF Staging:** [`papers/`](papers/) *(gitignored for `*.pdf`)*

---

## 1. Problem Framing & Two-Layer Design Philosophy

Every literature review in `docs/literature_survey/` must serve two audiences at once using a **Two-Layer Structure**:
1. **Top Half — Plain-English Intuition & Brainstorming (`Sections 0 & 1`):**
   - Modeled after the best parts of [`manual_reviews.md`](manual_reviews.md).
   - Written in clear, conversational English with **zero math symbols and zero unexplained paper buzzwords**.
   - Explains *what the trick actually is*, *how it compares to our project*, *what clever ideas we can steal*, and *where it might break on consumer hardware*.
   - Cross-checks [`manual_reviews.md`](manual_reviews.md) for team context, **but verifies every claim against the ground-truth PDF** to catch and correct any human skimming errors.
2. **Bottom Half — Technical & Engineering Reference (`Sections 2 to 5`):**
   - Provides the exact formulas (with every symbol explained in plain words right below the equation), data structures, empirical numbers from the paper's tables, and `kv_bench` implementation feasibility on our 8–16 GB consumer GPUs.

---

## 2. The 3-Tier Literature Taxonomy & Target Directories

- **Tier 1 — Systems & Architecture Foundations:** `docs/literature_survey/foundations/<slug>.md`
- **Tier 2 — KV Cache Optimization Techniques (Core Focus):** `docs/literature_survey/techniques/<slug>.md`
- **Tier 3 — Benchmarks, Datasets & Evaluation Metrics:** `docs/literature_survey/benchmarks/<slug>.md`

All templates and step-by-step execution rules are maintained in [`.agents/skills/literature-review/SKILL.md`](.agents/skills/literature-review/SKILL.md).
