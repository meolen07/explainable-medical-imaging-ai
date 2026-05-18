# Contributing to MedXAI

Thank you for your interest in **MedXAI: Explainable Medical Imaging Intelligence System** ([explainable-medical-imaging-ai](https://github.com/meolen07/explainable-medical-imaging-ai)). This project is maintained as open research and engineering software. Thoughtful contributions—whether code, documentation, tests, or reproducible experiments—help make medical imaging models more transparent and auditable.

We welcome contributors from academia, industry, and independent research. Please read this guide before opening a pull request.

---

## Code of Conduct

Participate in a professional, respectful, and inclusive manner:

- Treat others with courtesy; assume good intent in technical discussion.
- Focus feedback on ideas, code, and evidence—not individuals.
- Do not tolerate harassment, discrimination, or personal attacks.
- Escalate serious concerns to the maintainer via GitHub issues or direct contact on the repository profile.

By contributing, you agree to uphold these expectations.

---

## What to Contribute

MedXAI spans training, inference, explainability, API, and UI layers. Useful contributions include:

| Area | Examples |
|------|----------|
| **Bugs** | Incorrect tensor shapes, broken API routes, config parsing errors |
| **Documentation** | README clarifications, API examples, config comments, tutorials |
| **Models** | New backbones, ensemble strategies, calibration, export (ONNX) |
| **Tests** | Unit tests for metrics, Grad-CAM / attention map shapes, API contracts |
| **XAI** | Grad-CAM variants, ViT attention rollout, overlay quality, evaluation hooks |
| **Infrastructure** | Docker, CI workflows, dependency pinning, reproducibility fixes |

Before large changes (new model family, API redesign, dataset loaders), open an issue to align on scope and design.

---

## Medical and Research Disclaimer for Contributors

MedXAI is a **research and engineering reference**. It is **not** a medical device and is **not** validated for clinical diagnosis, treatment, or triage.

**By contributing, you agree that:**

- You will **not** submit protected health information (PHI), patient-identifiable data, or credentials.
- You will use only **de-identified** or **synthetic/public** data compliant with applicable policies (IRB, HIPAA, GDPR, or equivalent).
- You will **not** imply clinical efficacy, regulatory clearance, or diagnostic accuracy in docs, commit messages, or PR descriptions unless supported by rigorous, cited evidence—and even then, the project remains non-clinical.
- Explainability outputs are **heuristic attributions**, not ground-truth segmentations; document limitations when adding new XAI methods.

If you are unsure whether data or claims are appropriate, ask in an issue before contributing.

---

## Development Setup

### Prerequisites

- Python **3.10+**
- CUDA-capable GPU recommended for training (CPU acceptable for smoke tests)
- Git

### Environment

```bash
git clone https://github.com/meolen07/explainable-medical-imaging-ai.git
cd explainable-medical-imaging-ai

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Configuration and data

- Experiment and path settings live under `configs/` (e.g. `configs/data.yaml`, `configs/train_*.yaml`, `configs/inference.yaml`).
- Place local datasets under `data/` following the layout described in the README and `configs/data.yaml`. **Do not commit raw patient data.**
- Checkpoints (`checkpoints/`), runs (`runs/`), and outputs (`outputs/`) are typically gitignored.

### Smoke checks

```bash
# Run tests (see Testing below)
pytest

# Optional: start API and UI locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run src/ui/app.py
```

---

## Branch Naming and Pull Requests

### Branches

Use short, descriptive branch names:

- `feat/grad-cam-guided-variant`
- `fix/api-predict-shape`
- `docs/contributing-clarify-phi`
- `test/attention-rollout-shapes`

Avoid committing directly to `main` unless you are a maintainer performing a release fix.

### Pull request guidelines

1. **One logical change per PR** when possible (feature, fix, or doc set).
2. Link related issues (`Fixes #12`, `Relates to #8`).
3. Include a clear summary: **what** changed and **why**.
4. List **test commands** run and their outcome.
5. Add screenshots or example outputs for UI, API, or explanation overlay changes.
6. Update `configs/`, README, or docstrings when behavior or interfaces change.
7. Ensure CI passes (or note why checks are skipped).

Draft PRs are welcome for early feedback.

---

## Code Style

- **Language:** Python 3.10+ for core logic; type hints are **encouraged** on public functions and new modules.
- **Formatting:** Prefer consistent style with [Black](https://github.com/psf/black) (line length 88–100 per project config) and [Ruff](https://docs.astral.sh/ruff/) for linting and import sorting, if configured in `pyproject.toml` or CI.
- **Structure:** Follow existing layout under `src/` (`data/`, `models/`, `explain/`, `api/`, `ui/`).
- **Configs:** Prefer YAML-driven experiments over hard-coded hyperparameters.
- **Naming:** Use clear, domain-neutral names (`macro_f1`, `attention_rollout`); avoid clinical claims in identifiers.
- **Dependencies:** Minimize new dependencies; justify additions in the PR description.

When in doubt, match the style of the file you are editing.

---

## Testing Expectations

Tests live under `tests/` and are run with **pytest**.

```bash
# Full suite
pytest

# Verbose / single file
pytest -v tests/test_explain.py
```

**Before opening a PR:**

- Run `pytest` locally and ensure relevant tests pass.
- Add or update tests for new behavior (metrics, tensor shapes, API schemas, config loading).
- Do not weaken tests to make failing code pass without maintainer agreement.

GPU-only tests should be skipped or marked appropriately when CI has no GPU.

---

## Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/) for readable history and changelog-friendly summaries:

```
<type>(<optional scope>): <short description>

[optional body]

[optional footer: Fixes #123]
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`

**Examples:**

```
feat(explain): add guided Grad-CAM option for ResNet
fix(api): validate image MIME type on /predict
docs(readme): clarify non-clinical disclaimer
test(explain): assert Grad-CAM output matches input spatial size
```

Use the imperative mood in the subject line (`add` not `added`). Keep the subject ≤ 72 characters when practical.

---

## Reporting Issues

Search [existing issues](https://github.com/meolen07/explainable-medical-imaging-ai/issues) before filing a duplicate.

Include:

1. **Summary** — expected vs actual behavior.
2. **Reproduction steps** — minimal commands or code path.
3. **Environment** — OS, Python version, PyTorch/CUDA, GPU model (if relevant).
4. **Config** — relevant `configs/*.yaml` snippets (redact paths if needed).
5. **Logs / stack traces** — in fenced code blocks.
6. **Screenshots** — for UI or heatmap issues (use synthetic or public examples only).

For security-sensitive reports (e.g. accidental secret commit), contact the maintainer privately rather than opening a public issue.

---

## Review Process

1. A maintainer or collaborator will triage new issues and PRs.
2. Reviewers may request changes, tests, or documentation updates.
3. Approval requires passing checks and alignment with project scope (research-grade XAI, not clinical product claims).
4. Maintainers may squash-merge or rebase-merge depending on commit history; contributors do not need to force-push unless asked.

Constructive review comments should cite reproducible evidence (failing test, benchmark note, paper reference).

---

## License

By contributing code, documentation, or other materials to this repository, you agree that your contributions will be licensed under the same terms as the project: the **[MIT License](LICENSE)**.

You retain copyright to your contributions. The project copyright is held by **Huynh Mai Linh Nguyen** (see `LICENSE`).

---

## Questions

- **Repository:** https://github.com/meolen07/explainable-medical-imaging-ai  
- **Maintainer:** [@meolen07](https://github.com/meolen07)

For process questions, open a GitHub issue labeled `question`.

---

<sub>Mọi đóng góp đều được hoan nghênh; vui lòng tuân thủ quy định về dữ liệu y tế và không sử dụng phần mềm cho mục đích lâm sàng. — Contributions are welcome; please follow medical-data rules and do not use this software for clinical purposes.</sub>
