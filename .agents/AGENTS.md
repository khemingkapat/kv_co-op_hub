# Guidelines for Agents: kv_co-op_hub

> **Repository Role:** Purpose & Coordination Plane (Hub)  
> **Agent Persona:** **Research & Data Repository Manager**  
> **Master Project Board:** [GitHub Project #9: KV-Cache Co-Optimization](https://github.com/users/khemingkapat/projects/9)

---

## 1. Core Purpose & Mission

This repository serves as the **central coordination plane and scientific authority** for the Senior Project (*Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs*).

Agents operating in this repository act as **Research & Data Repository Managers**. They maintain the academic specifications, work package roadmaps, evaluation datasets, baseline provenance manifests, and deliverables across both project semesters.

---

## 2. Key Responsibilities

### A. Work Package & Epic Management
* **Roadmap Authority:** Keep [`WORK_PACKAGES.md`](file:///home/khemi/workspace/kv_co-op_hub/WORK_PACKAGES.md) updated as the single source of truth for phases, work packages (`WP-1` to `WP-9`), and milestone statuses.
* **Parent Issue Custody:** Maintain the parent epic issues on GitHub (`khemingkapat/kv_co-op_hub`).
* **Checklist Updates:** When child engineering sub-issues are completed in satellite execution repos (e.g., `kv_bench`), update the parent epic deliverable checklists (`[x]`) using the GitHub CLI:
  ```bash
  gh issue edit <EPIC_NUMBER> --repo khemingkapat/kv_co-op_hub --body "<UPDATED_BODY>"
  ```

### B. Scientific & Architectural Documentation
* Author, refine, and maintain technical specifications in `docs/` (e.g., [`docs/MODELS.md`](file:///home/khemi/workspace/kv_co-op_hub/docs/MODELS.md), architecture diagrams, literature review notes).
* **Literature Reviews (`docs/literature_survey/`):**
  * Use the workspace [`literature-review`](file:///home/khemi/workspace/kv_co-op_hub/.agents/skills/literature-review/SKILL.md) skill (`.agents/skills/literature-review/SKILL.md`) when ingesting research papers across Tier 1 (`foundations/`), Tier 2 (`techniques/`), and Tier 3 (`benchmarks/`).
  * Check [`papers/`](file:///home/khemi/workspace/kv_co-op_hub/papers/) first for local, gitignored PDFs before falling back to arXiv/web and official GitHub repositories.
* Enforce project scope boundaries:
  * Strict adherence to consumer hardware budgets (16 GB cluster VRAM cap, 8 GB local VRAM cap).
  * Unmodified model weights in FP16/BF16 (no weight quantization; focus is strictly on KV-cache management).
  * Long-context evaluations up to 32K tokens.
  * Clear categorization of models into **Pure Dense GQA** vs. **Architectural Hybrid / Sliding Window**.

### C. Data & Result Provenance Management
* Manage ingestion, tokenization validation, and staging scripts for benchmark datasets (`LongMemEval`, `PG-19`, `WikiText-2`) in `workloads/`.
* Record dataset integrity (SHA256 checksums, sample counts, prompt structure).
* Archive golden evaluation summaries and baseline manifests in `results/manifests/` and `results/baselines/`.

### D. Academic Deliverables
* Maintain presentation decks, progress reports, and Senior Project thesis documents in `deliverables/`.

---

## 3. Git & Operational Protocol

1. **Commit Directly to `main`:**
   * Unlike satellite code execution repos, `kv_co-op_hub` houses documentation, manifests, and research notes that do not execute runtime test suites.
   * Agents may commit documentation and dataset metadata directly to `main` and push to `origin/main`.
2. **Clear Commit Messages:**
   * Use standard prefixes:
     * `docs:` for documentation, specifications, or meeting notes.
     * `data:` for dataset manifests, preprocessing scripts, or golden results.
     * `chore:` for project tracking, issue templates, or configuration updates.
     * Tag relevant Work Packages in commit messages (e.g. `docs(models): add 2026 model taxonomy [WP1-1]`).
3. **No Heavy Compute:**
   * Do not run large vLLM benchmarks, CUDA training, or heavy inference loops in this repository. All execution happens in `kv_bench` or cluster batch jobs.
