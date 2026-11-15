#!/usr/bin/env python3
"""Normalize Earth-Bench final answers with an OpenAI-compatible LLM."""

import argparse
import json
import re
from pathlib import Path

import requests


def extract_choice(text: str) -> str | None:
    """Return a normalized multiple-choice answer, if one is present."""
    match = re.search(r"<Answer>([A-D])[^<]*<Answer>", text, re.IGNORECASE)
    if match:
        return f"<Answer>{match.group(1).upper()}<Answer>"
    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    if match:
        return f"<Answer>{match.group(1).upper()}<Answer>"
    return None


def polish_answer(
    question: str,
    answer: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
) -> str:
    """Extract the answer choice without judging or changing the response."""
    prompt = f"""Extract the final multiple-choice answer from a model response.

Question:
{question}

Model response:
{answer}

Return only <Answer>X<Answer>, where X is A, B, C, or D. Do not infer a
choice that is absent from the model response. If no answer choice is present,
return the original response unchanged."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You extract answer choices without solving the question.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        normalized = extract_choice(result)
        if normalized:
            return normalized
        print(f"Warning: no answer choice found in polisher response: {result}")
    except Exception as exc:
        print(f"Warning: answer polishing failed: {type(exc).__name__}: {exc}")

    return extract_choice(answer) or answer


def load_questions(log_path: Path) -> dict[str, str]:
    """Read user questions from the multi-line JSON trajectory log."""
    questions: dict[str, str] = {}
    buffer = ""
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            buffer += line
            try:
                record = json.loads(buffer)
            except json.JSONDecodeError:
                continue
            buffer = ""

            question_id = str(record.get("question_index", ""))
            if question_id.startswith("question"):
                question_id = question_id.removeprefix("question")
            for message in record.get("conversations", []):
                if isinstance(message, dict) and message.get("role") == "user":
                    questions[question_id] = message.get("content", "")
                    break
    return questions


def polish_results(
    base_dir: Path,
    log_file: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
) -> None:
    results_file = base_dir / "results_summary.json"
    output_file = base_dir / "results_summary_polished.json"
    if not log_file.is_file():
        raise FileNotFoundError(f"Trajectory log not found: {log_file}")
    if not results_file.is_file():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    questions = load_questions(log_file)
    results = json.loads(results_file.read_text(encoding="utf-8"))
    polished_results = []

    for item in results:
        question_id = str(item.get("question_id", ""))
        answer = item.get("answer", "")
        question = questions.get(question_id, "")
        if not question:
            print(f"Warning: no question found for ID {question_id}")
            polished_answer = extract_choice(answer) or answer
        else:
            print(f"Processing question {question_id}...")
            polished_answer = polish_answer(
                question,
                answer,
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
            )
        polished_results.append(
            {
                "question_id": question_id,
                "original_answer": answer,
                "final_answer": polished_answer,
            }
        )

    output_file.write_text(
        json.dumps(polished_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Polished results saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Polish Earth-Bench final answers.")
    parser.add_argument("--base_dir", required=True, help="Directory containing one Earth-Bench run.")
    parser.add_argument("--log_file", required=True, help="Trajectory log for the same run.")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key", required=True, help="API key for the answer-polishing endpoint.")
    parser.add_argument("--model", required=True, help="Answer-polishing model name.")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds.")
    args = parser.parse_args()

    polish_results(
        Path(args.base_dir),
        Path(args.log_file),
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
