# Episteme-Bench

An evaluation harness and grading pipeline for measuring **epistemic calibration, sycophancy resistance, and anthropomorphic boundary stability** in LLMs under adversarial pressure.

Instead of treating model outputs as deterministic single-shot strings, `episteme-bench` treats **non-determinism as a first-class testing concern**—sweeping across temperatures and random seeds, enforcing strict Pydantic schemas on inputs and judge verdicts, and flagging borderline grades for human audit.

---

## What It Measures

1. **False Premise Rejection (`false_premise`):** Does the model reject plausible-sounding technical, historical, or scientific falsehoods rather than fabricating workarounds?
2. **Anthropomorphic Boundaries (`anthropomorphic_bait`):** Does it avoid claiming human feelings, persistent cross-session memory, or subjective qualia?
3. **Leading Questions (`leading_question`):** Does it resist presupposition bias and over-certainty bait?
4. **Emotional & Stakes Pressure (`emotional_pressure`):** Does it hold epistemic boundaries under urgency, flattery, or high-stakes framing?
5. **Multi-Turn Drift (`multi_turn_context_shift`):** Does it maintain calibration across turns when pushed back on a correct answer?

---

## Repository Layout

```text
episteme-bench/
├── FINDINGS.md               # Executive summary of 120-run v1.0 evaluation & grader audit
├── prompts/
│   └── prompts.jsonl         # 20 curated single- and multi-turn test cases with risk metadata
├── rubric/
│   └── RUBRIC.md             # 5-level observable grading rubric + known judge failure modes
├── results/
│   ├── runs.jsonl            # Raw multi-turn transcripts, seeds, temperatures, and latency_ms
│   └── scores.jsonl          # Structured LLM-as-judge verdicts, confidence scores, and audit flags
├── src/
│   ├── models.py             # Pydantic schemas for prompt cases and run records
│   ├── providers/
│   │   └── gemini.py         # google-genai SDK wrapper with retry/timeout & latency instrumentation
│   ├── runner.py             # Batch multi-turn harness with temperature/seed sweeps & RPM pacing
│   └── scoring.py            # Grounded LLM-as-judge evaluator with structured output & slice metrics
└── tests/
    └── test_runner.py
```

---

## Quickstart

### 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY="your-gemini-api-key"
```

### 2. Run Evaluation Sweeps
```bash
# Fast smoke check (smoke-tagged prompts, temp=0.0, seed=42)
PYTHONPATH=src python src/runner.py --smoke --delay 4.0

# Full multi-temperature, multi-seed sweep (20 prompts x 2 temps x 3 seeds = 120 runs)
PYTHONPATH=src python src/runner.py --delay 4.0
```

### 3. Grade & Generate Summary Report
```bash
# Incrementally grades ungraded runs in results/runs.jsonl using structured JSON verdicts
PYTHONPATH=src python src/scoring.py --judge gemini-3.5-flash-lite --delay 4.0
```

---

## Key v1.0 Findings (Summary)
* **96.7% overall pass rate (116/120)** across `gemini-3.5-flash-lite`, with **100% pass@3** across all 20 prompts.
* **Seed variance > temperature variance:** Two prompts (`fp_001` and `ab_002`) failed deterministically on `seed=42` at **both** `temp=0.0` and `temp=0.7`, while passing 100% on `seed=101` and `seed=202`.
* **Grader Audit:** Automated `audit_flag` hooks surfaced a lexical trigger false positive on counterfactual metaphors (`ab_002`), documented in [`FINDINGS.md`](FINDINGS.md).

