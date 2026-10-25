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
  * **Hardware Capacity Specification:** Strict adherence to consumer hardware budgets (16 GB cluster VRAM cap on RTX 4090, 8 GB local VRAM cap on RTX 4060).
  * **Controlled Base Model vs. `STRUCTURAL` KV-Cache Compression:** Base model checkpoints must match the pinned BF16/FP16 versions in [`docs/MODELS.md`](file:///home/khemi/workspace/kv_co-op_hub/docs/MODELS.md) (no arbitrary pre-quantized GGUF/AWQ weights), while KV-cache quantization (FP8, INT4, TurboQuant, HeadQ) and token eviction are evaluated under the `STRUCTURAL` optimization axis.
  * **Two Primary Quality Pillars:** Production quality evaluation focuses exclusively on (1) **Micro-Level Logit Divergence** (`top1_agreement`, `kl_div_mean`, `kl_div_p99`) and (2) **Macro-Level LongMemEval** (5 core memory capabilities). Perplexity ($\Delta\text{PPL}$) is retained strictly as an optional supplementary diagnostic.
  * Long-context evaluations up to 32K tokens, categorizing models into **Pure Dense GQA** vs. **Architectural Hybrid / Sliding Window**.

### C. Data & Result Provenance Management
* Manage ingestion, tokenization validation, and staging scripts for benchmark datasets (`LongMemEval`, `PG-19`) in `workloads/`.
* Record dataset integrity (`results/manifests/dataset_manifest.json`) and run provenance (`results/manifests/run_manifest_*.json` linking Slurm Job ID, hardware specs, `kv_bench` commit SHA, and SHA256 checksums).
* Archive consolidated golden evaluation summaries in `results/baselines/` and `results/runs/`.
* **Logprob Bloat Guardrail:** Raw per-step token logprob reference files (`*_ref.json`) are restricted strictly to `results/baselines/references/`. Candidate and experimental sweeps must record only computed summary metrics (`top1_agreement`, `kl_div_mean`, `kl_div_p99`).

### D. Academic Deliverables
* Maintain presentation decks, progress reports, and Senior Project thesis documents in `deliverables/`.

---

## 3. Git & Operational Protocol

1. **Branch & Pull Request Workflow (No Direct Push to `main`):**
   * Because `kv_co-op_hub` serves as the central scientific data repository across multiple collaborators, **all changes (human or agent) must go through a feature branch (`data/*`, `docs/*`, `feat/*`, `chore/*`) and a Pull Request targeting `main`**.
   * Every PR must complete the checklist in [`.github/pull_request_template.md`](file:///home/khemi/workspace/kv_co-op_hub/.github/pull_request_template.md), pass automated CI checks ([`.github/workflows/hub-ci.yml`](file:///home/khemi/workspace/kv_co-op_hub/.github/workflows/hub-ci.yml)), and receive approval from `@khemingkapat` ([`.github/CODEOWNERS`](file:///home/khemi/workspace/kv_co-op_hub/.github/CODEOWNERS)).
2. **Clear Commit Messages:**
   * Use standard prefixes:
     * `docs:` for documentation, specifications, or meeting notes.
     * `data:` for dataset manifests, preprocessing scripts, or golden results.
     * `chore:` for project tracking, issue templates, CI workflows, or configuration updates.
     * Tag relevant Work Packages in commit messages (e.g. `data(baselines): stage WP-2 golden baseline and run manifest [WP-2]`).
3. **No Heavy Compute or Binary Bloat:**
   * Do not run large vLLM benchmarks, CUDA training, or heavy inference loops in this repository. All execution happens in `kv_bench` or cluster batch jobs.
   * Never commit binary checkpoints (`.pt`, `.bin`, `.safetensors`, `.gguf`), raw Slurm `.log` files, or files exceeding 10 MB.
