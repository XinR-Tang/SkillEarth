import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON file data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_tool_names_from_calls(tool_calls: List[Dict]) -> List[str]:
    """Extract tool names from tool calls list"""
    return [call.get("name", "") for call in tool_calls]

def find_tool_calls_from_data(data: Dict) -> List[str]:
    """Extract tool call name list from data"""
    tool_calls = data.get("tool_calls", [])
    return extract_tool_names_from_calls(tool_calls)

def contains_all_tool_calls_any_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Check if all expected tool calls are contained (order not considered) - soft scoring version
    """
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)
    
    if not expected_tools:
        return {
            "score": 1.0,
            "key": "contains_all_tool_calls_any_order",
            "expected": expected_tools,
            "actual": actual_tools,
            "details": {"matched_tools": 0, "total_expected": 0}
        }
    
    # Calculate the number of matched tools
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    matched_tools = expected_set.intersection(actual_set)
    
    score = len(matched_tools) / len(expected_set)
    
    return {
        "score": score, 
        "key": "contains_all_tool_calls_any_order",
        "expected": expected_tools,
        "actual": actual_tools,
        "details": {
            "matched_tools": len(matched_tools),
            "total_expected": len(expected_set),
            "matched_tool_names": list(matched_tools)
        }
    }

def contains_all_tool_calls_in_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Check if all expected tool calls are contained (considering order, but allowing other tools in between) - soft scoring version
    """
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)
    
    if not expected_tools:
        return {
            "score": 1.0,
            "key": "contains_all_tool_calls_in_order",
            "expected": expected_tools,
            "actual": actual_tools,
            "details": {"matched_in_order": 0, "total_expected": 0}
        }
    
    # Calculate the number of tools matched in order
    matched_count = 0
    actual_iter = iter(actual_tools)
    
    for expected_tool in expected_tools:
        # Try to find this tool in the remaining actual_tools
        found = False
        for actual_tool in actual_iter:
            if actual_tool == expected_tool:
                matched_count += 1
                found = True
                break
        if not found:
            # If not found, break the loop (strict order requirement)
            break
    
    score = matched_count / len(expected_tools)
    
    return {
        "score": score, 
        "key": "contains_all_tool_calls_in_order",
        "expected": expected_tools,
        "actual": actual_tools,
        "details": {
            "matched_in_order": matched_count,
            "total_expected": len(expected_tools)
        }
    }

def trajectory_step_wise_score(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Calculate trajectory step-wise score (strict order matching)
    """
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)

    if not expected_tools:
        return {"score": 0, "key": "trajectory_step_wise"}

    # Calculate the number of correctly matched steps
    correct_steps = 0
    min_length = min(len(expected_tools), len(actual_tools))

    for i in range(min_length):
        if expected_tools[i] == actual_tools[i]:
            correct_steps += 1
        else:
            break  # Stop as soon as there's a mismatch (strict order)

    score = correct_steps / len(expected_tools) if len(expected_tools) > 0 else 0

    return {
        "score": score,
        "key": "trajectory_step_wise",
        "details": {
            "correct_steps": correct_steps,
            "total_expected": len(expected_tools),
            "expected_sequence": expected_tools,
            "actual_sequence": actual_tools[:len(expected_tools)]
        }
    }

EARTHBENCH_PATH_MARKER = "benchmark/data/"


def normalize_parameter_value(value: Any) -> Any:
    """Normalize equivalent path representations without changing other values."""
    if isinstance(value, str):
        normalized = value.replace("\\\\", "/")
        marker_index = normalized.find(EARTHBENCH_PATH_MARKER)
        if marker_index >= 0:
            return normalized[marker_index:]
        return normalized
    if isinstance(value, dict):
        return {key: normalize_parameter_value(item) for key, item in value.items()}
    if isinstance(value, list):
        # Keep list order: batch tool parameters must remain aligned by index.
        return [normalize_parameter_value(item) for item in value]
    return value


def parameter_similarity(predicted_value: Any, expected_value: Any) -> float:
    """Return a semantic parameter score in [0, 1].

    Expected dictionary keys and list positions are scored independently. Extra
    predicted dictionary keys are ignored, while missing expected values score 0.
    """
    predicted_value = normalize_parameter_value(predicted_value)
    expected_value = normalize_parameter_value(expected_value)

    if isinstance(expected_value, dict):
        if not isinstance(predicted_value, dict):
            return 0.0
        if not expected_value:
            return 1.0
        return sum(
            parameter_similarity(predicted_value.get(key), expected_item)
            if key in predicted_value else 0.0
            for key, expected_item in expected_value.items()
        ) / len(expected_value)

    if isinstance(expected_value, list):
        if not isinstance(predicted_value, list):
            return 0.0
        if not expected_value:
            return 1.0
        return sum(
            parameter_similarity(predicted_value[index], expected_item)
            if index < len(predicted_value) else 0.0
            for index, expected_item in enumerate(expected_value)
        ) / len(expected_value)

    if (isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool)
            and isinstance(predicted_value, (int, float)) and not isinstance(predicted_value, bool)):
        tolerance = 1e-6 * max(1.0, abs(float(expected_value)))
        return 1.0 if abs(float(predicted_value) - float(expected_value)) <= tolerance else 0.0

    return 1.0 if predicted_value == expected_value else 0.0


def check_parameter_accuracy(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """Evaluate aligned tool-call parameters with path normalization and partial credit."""
    predicted_calls = predicted_data.get("tool_calls", [])
    expected_calls = ground_truth_data.get("tool_calls", [])

    if not expected_calls:
        return {
            "score": 1.0,
            "key": "parameter_accuracy",
            "details": {
                "reason": "No expected tool calls",
                "expected_count": 0,
                "actual_count": len(predicted_calls),
                "matched_steps": 0,
                "total_expected_steps": 0,
            },
        }

    total_expected_steps = len(expected_calls)
    matched_steps = 0
    step_scores = []
    parameter_details = []

    for index, expected_call in enumerate(expected_calls):
        step = index + 1
        if index >= len(predicted_calls):
            parameter_details.append({
                "step": step,
                "expected_tool_name": expected_call.get("name", ""),
                "actual_tool_name": None,
                "name_match": False,
                "input_match": False,
                "parameter_score": 0.0,
                "is_correct": False,
                "reason": "Missing step",
            })
            step_scores.append(0.0)
            continue

        predicted_call = predicted_calls[index]
        name_match = predicted_call.get("name") == expected_call.get("name")
        parameter_score = (
            parameter_similarity(predicted_call.get("input", {}), expected_call.get("input", {}))
            if name_match else 0.0
        )
        input_match = parameter_score == 1.0
        if name_match and input_match:
            matched_steps += 1

        parameter_details.append({
            "step": step,
            "expected_tool_name": expected_call.get("name", ""),
            "actual_tool_name": predicted_call.get("name", ""),
            "name_match": name_match,
            "input_match": input_match,
            "parameter_score": parameter_score,
            "is_correct": name_match and input_match,
        })
        step_scores.append(parameter_score)

    score = sum(step_scores) / total_expected_steps
    return {
        "score": score,
        "key": "parameter_accuracy",
        "details": {
            "matched_steps": matched_steps,
            "total_expected_steps": total_expected_steps,
            "expected_count": len(expected_calls),
            "actual_count": len(predicted_calls),
            "call_details": parameter_details,
            "scoring_method": "aligned_semantic_partial_credit",
            "path_normalization": EARTHBENCH_PATH_MARKER,
        },
    }
def evaluate_single_question(predicted_data: Dict, ground_truth_data: Dict) -> Dict:
    """
    Perform complete evaluation for a single question
    """
    results = {}

    # Execute evaluation metrics (removed exact_match)
    results["contains_all_tool_calls_any_order"] = contains_all_tool_calls_any_order(predicted_data, ground_truth_data)
    results["contains_all_tool_calls_in_order"] = contains_all_tool_calls_in_order(predicted_data, ground_truth_data)
    results["trajectory_step_wise_score"] = trajectory_step_wise_score(predicted_data, ground_truth_data)
    results["parameter_accuracy"] = check_parameter_accuracy(predicted_data, ground_truth_data)

    return results

def run_step_by_step_evaluation(predicted_file: str, ground_truth_file: str) -> Dict:
    """
    Run complete step-by-step evaluation
    """
    # Load data
    predicted_data = load_json_data(predicted_file)
    ground_truth_data = load_json_data(ground_truth_file)
    
    # Create index mapping (based on question_index)
    # Handle different index formats: ground truth uses "question1", prediction uses "1"
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {}
    for item in predicted_data:
        key = item["question_index"]
        # If the prediction data key is a pure number, convert to "questionX" format
        if key.isdigit():
            key = f"question{key}"
        pred_dict[key] = item
    
    all_results = {}
    summary_stats = {
        "total_questions": 0,
        "evaluated_questions": 0,
        "missing_predictions": [],
        "metrics_summary": {
            "contains_all_tool_calls_any_order": {"total_score": 0, "count": 0},
            "contains_all_tool_calls_in_order": {"total_score": 0, "count": 0},
            "trajectory_step_wise_score": {"total_score": 0, "count": 0},
            "parameter_accuracy": {"total_score": 0, "count": 0}
        }
    }
    
    # Score every item in the supplied ground-truth split.  Missing or malformed
    # trajectories are empty predictions and therefore receive zero credit; they
    # must not be excluded from the test-set denominator.
    for question_index, gt_item in gt_dict.items():
        summary_stats["total_questions"] += 1

        if question_index not in pred_dict:
            summary_stats["missing_predictions"].append(question_index)
            pred_item = {"question_index": question_index, "tool_calls": []}
        else:
            pred_item = pred_dict[question_index]
        summary_stats["evaluated_questions"] += 1
        
        # Evaluate single question
        question_results = evaluate_single_question(pred_item, gt_item)
        all_results[question_index] = question_results
        
        # Update summary statistics
        for metric_name, metric_result in question_results.items():
            if metric_name in summary_stats["metrics_summary"]:
                summary_stats["metrics_summary"][metric_name]["total_score"] += metric_result["score"]
                summary_stats["metrics_summary"][metric_name]["count"] += 1
    
    # Calculate average scores
    for metric_name, metric_stats in summary_stats["metrics_summary"].items():
        if metric_stats["count"] > 0:
            metric_stats["average_score"] = metric_stats["total_score"] / metric_stats["count"]
        else:
            metric_stats["average_score"] = 0
    
    return {
        "individual_results": all_results,
        "summary": summary_stats
    }

def print_evaluation_summary(results: Dict, model_name: str = ""):
    """
    Print evaluation results summary
    """
    summary = results["summary"]
    print("=" * 60)
    if model_name:
        print(f"Step-by-Step Evaluation Results Summary - {model_name}")
    else:
        print("Step-by-Step Evaluation Results Summary")
    print("=" * 60)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Evaluated questions: {summary['evaluated_questions']}")
    print(f"Missing predictions: {len(summary['missing_predictions'])}")

    if summary['missing_predictions']:
        print(f"Missing questions: {summary['missing_predictions']}")

    print("\nAverage scores for each metric:")
    print("-" * 60)
    for metric_name, metric_stats in summary["metrics_summary"].items():
        avg_score = metric_stats.get("average_score", 0)
        count = metric_stats.get("count", 0)
        print(f"{metric_name}: {avg_score:.4f} (based on {count} questions)")

    print("=" * 60)


def main(args):
    """Evaluate a single model directory and save results in the same directory."""

    model_dir = Path(args.model_dir).expanduser().resolve()
    ground_truth_file = Path(args.ground_truth_file).expanduser().resolve()

    predicted_file = model_dir / "extracted_tool_calls.json"
    output_file = model_dir / "step_by_step_evaluation_results.json"

    print("Starting step-by-step evaluation")
    print(f"Model directory: {model_dir}")
    print(f"Prediction file: {predicted_file}")
    print(f"Ground truth file: {ground_truth_file}")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory not found: {model_dir}")

    if not predicted_file.is_file():
        raise FileNotFoundError(f"Prediction file not found: {predicted_file}")

    if not ground_truth_file.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    # Run evaluation
    results = run_step_by_step_evaluation(
        str(predicted_file),
        str(ground_truth_file)
    )

    # Print summary
    print_evaluation_summary(results, model_dir.name)

    # Save results to the specified model directory
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation completed")
    print(f"Results saved to: {output_file}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-by-Step evaluation for earth benchmark")
    parser.add_argument("--model_dir", type=str, default="evaluate_langchain", help="Directory containing extracted_tool_calls.json and where evaluation results will be saved.")
    parser.add_argument("--ground_truth_file", type=str, default="extracted_tool_calls_GT.json", help="Path to the ground-truth tool-call JSON file.")
    args = parser.parse_args()
    main(args)
