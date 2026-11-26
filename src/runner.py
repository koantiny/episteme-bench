import argparse
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from google.genai import types

from models import PromptCase, EvalRunRecord
from providers.gemini import call_gemini

PROMPTS_FILE = Path("prompts/prompts.jsonl")
RESULTS_FILE = Path("results/runs.jsonl")


def load_prompts(path: Path) -> list[PromptCase]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(PromptCase.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"Schema error in {path} at line {line_num}: {e}")
    return cases


def execute_case(
    case: PromptCase,
    model: str,
    temperature: float,
    seed: int,
    run_id: str,
) -> EvalRunRecord:
    contents: list[types.Content] = []
    transcript: list[dict[str, str]] = []
    total_latency_ms = 0.0
    last_response = ""

    for user_turn in case.turns:
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_turn)])
        )
        transcript.append({"role": "user", "content": user_turn})

        reply, latency = call_gemini(
            contents=contents,
            model=model,
            temperature=temperature,
            seed=seed,
        )
        total_latency_ms += latency
        last_response = reply

        contents.append(
            types.Content(role="model", parts=[types.Part.from_text(text=reply)])
        )
        transcript.append({"role": "model", "content": reply})

    return EvalRunRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_id=case.id,
        category=case.category,
        expected_risk=case.expected_risk,
        tags=case.tags,
        model=model,
        temperature=temperature,
        seed=seed,
        latency_ms=round(total_latency_ms, 2),
        turns_transcript=transcript,
        final_response=last_response,
    )


def run_harness(
    temperatures: list[float] | None = None,
    seeds: list[int] | None = None,
    model: str = "gemini-3.5-flash-lite",
    tag_filter: str | None = None,
    delay: float = 4.0,
):
    if temperatures is None:
        temperatures = [0.0, 0.7]
    if seeds is None:
        seeds = list((42, 101, 202))

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    cases = load_prompts(PROMPTS_FILE)
    if tag_filter:
        cases = [c for c in cases if tag_filter in c.tags]

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    total_jobs = len(cases) * len(temperatures) * len(seeds)
    print(
        f"Starting {run_id}: {len(cases)} prompts x {len(temperatures)} temps x {len(seeds)} seeds = {total_jobs} runs"
    )

    completed = 0
    with RESULTS_FILE.open("a", encoding="utf-8") as out:
        for case in cases:
            for temp in temperatures:
                for seed in seeds:
                    record = execute_case(case, model, temp, seed, run_id)
                    out.write(record.model_dump_json() + "\n")
                    out.flush()
                    completed += 1
                    print(
                        f"[{completed}/{total_jobs}] {case.id} | temp={temp} | seed={seed} | {record.latency_ms}ms"
                    )
                    if delay > 0 and completed < total_jobs:
                        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 temp, 1 seed on 'smoke' tagged prompts",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds to sleep between runs (free-tier RPM pacing)",
    )
    args = parser.parse_args()

    if args.smoke:
        run_harness(temperatures=[0.0], seeds=list((42,)), tag_filter="smoke")
    else:
        run_harness()
