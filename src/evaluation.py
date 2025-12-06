"""
Evaluation and output functions for NTDS Complication Extractor.
Handles scoring, console output, and results file writing.
"""

import json
from datetime import datetime
import os


def compare_outputs_and_score(
    llm_outputs: dict,
    true_outputs: dict,
    mapping_dict: dict
) -> tuple[int, int]:
    """Compare LLM predictions against ground truth and calculate accuracy.

    Args:
        llm_outputs: Dict with complication labels as keys and predicted bools as values
        true_outputs: Dict with complication IDs as keys and actual bools as values
        mapping_dict: Dict mapping complication labels to IDs

    Returns:
        tuple: (num_correct, num_incorrect)
    """
    num_correct = 0
    num_incorrect = 0

    # Convert LLM outputs from labels to ids
    llm_output_with_ids = {mapping_dict[key]: val for key, val in llm_outputs.items()}

    for key in llm_output_with_ids:
        if true_outputs[key] == llm_output_with_ids[key]:
            num_correct += 1
        else:
            num_incorrect += 1

    return num_correct, num_incorrect


def generate_evaluation_report(
    llm_outputs: dict,
    true_outputs: dict,
    mapping_dict: dict
) -> dict:
    """Generate detailed per-complication evaluation report.

    Args:
        llm_outputs: Dict with complication labels as keys and predicted bools as values
        true_outputs: Dict with complication IDs as keys and actual bools as values
        mapping_dict: Dict mapping complication labels to IDs

    Returns:
        dict: Detailed evaluation report with per-complication results
    """
    # Convert LLM outputs from labels to ids
    llm_output_with_ids = {mapping_dict[key]: val for key, val in llm_outputs.items()}

    per_complication = []
    incorrect_predictions = []

    for label, complication_id in mapping_dict.items():
        if complication_id in llm_output_with_ids:
            predicted = llm_output_with_ids[complication_id]
            actual = true_outputs[complication_id]
            correct = (predicted == actual)

            per_complication.append({
                "id": complication_id,
                "label": label,
                "predicted": predicted,
                "actual": actual,
                "correct": correct
            })

            if not correct:
                incorrect_predictions.append({
                    "complication": label,
                    "predicted": "Yes" if predicted else "No",
                    "actual": "Yes" if actual else "No"
                })

    num_correct = sum(1 for item in per_complication if item["correct"])
    num_incorrect = len(per_complication) - num_correct
    total = len(per_complication)
    accuracy = num_correct / total if total > 0 else 0.0

    return {
        "summary": {
            "total_assessed": total,
            "correct": num_correct,
            "incorrect": num_incorrect,
            "accuracy": accuracy
        },
        "per_complication": per_complication,
        "incorrect_predictions": incorrect_predictions
    }


def print_console_summary(
    llm_outputs: dict,
    num_correct: int,
    num_incorrect: int,
    encounter_id: int,
    complications_info: list
) -> None:
    """Print formatted summary to console.

    Args:
        llm_outputs: Dict with complication labels and predictions
        num_correct: Number of correct predictions
        num_incorrect: Number of incorrect predictions
        encounter_id: The encounter ID being assessed
        complications_info: List of complication info dicts for ordering
    """
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    total = num_correct + num_incorrect
    accuracy = (num_correct / total * 100) if total > 0 else 0.0

    print(f"\nEncounter ID: {encounter_id}")
    print(f"Accuracy: {num_correct}/{total} ({accuracy:.1f}%)")
    print(f"\nCorrect Predictions: {num_correct}")
    print(f"Incorrect Predictions: {num_incorrect}")


def write_results_file(
    llm_outputs: dict,
    evaluation: dict,
    output_path: str,
    metadata: dict
) -> None:
    """Write detailed results to JSON file.

    Args:
        llm_outputs: Dict with complication labels and predictions
        evaluation: Evaluation report from generate_evaluation_report()
        output_path: Path to output JSON file
        metadata: Additional metadata to include (e.g., config)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build output structure
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "encounter_id": metadata.get("config", {}).get("encounter_id"),
            "complications_assessed": evaluation["summary"]["total_assessed"],
            "config": metadata.get("config", {})
        },
        "assessments": {},
        "evaluation": evaluation
    }

    # Add assessments
    for item in evaluation["per_complication"]:
        result["assessments"][item["id"]] = {
            "label": item["label"],
            "predicted": item["predicted"],
            "ground_truth": item["actual"],
            "correct": item["correct"]
        }

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n Results saved to: {output_path}")
