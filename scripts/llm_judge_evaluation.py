"""
LLM-as-a-Judge evaluation script for ECG generation quality assessment.
Samples 200 examples and uses Claude/GPT to evaluate generation quality.
"""

import json
import random
import os
from pathlib import Path
from typing import Dict, List, Any
import time
from tqdm import tqdm

# Try to import anthropic, fallback to openai if not available
try:
    import anthropic
    USE_ANTHROPIC = True
except ImportError:
    USE_ANTHROPIC = False
    try:
        import openai
        USE_OPENAI = True
    except ImportError:
        USE_OPENAI = False
        print("Warning: Neither anthropic nor openai library found. Please install one.")


def sample_examples(data: Dict, n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """Sample n examples from the full dataset."""
    random.seed(seed)

    # Convert dict to list of entries with keys
    all_entries = []
    for key, value in data.items():
        entry = value.copy()
        entry['entry_id'] = key
        all_entries.append(entry)

    # Sample randomly
    sampled = random.sample(all_entries, min(n_samples, len(all_entries)))

    print(f"Sampled {len(sampled)} examples from {len(all_entries)} total entries")
    return sampled


def create_judge_prompt(question: str, ground_truth: str, generation: str) -> str:
    """Create evaluation prompt for LLM judge."""

    prompt = f"""You are an expert cardiologist and AI evaluator. Your task is to evaluate the quality of an AI-generated ECG interpretation compared to the ground truth expert interpretation.

**Question Asked:**
{question}

**Ground Truth (Expert Interpretation):**
{ground_truth}

**AI Generated Response:**
{generation}

Please evaluate the AI response on the following criteria (score each 1-5):

1. **Accuracy**: Does the AI response align with the ground truth? Are the key findings correct?
2. **Completeness**: Does it capture all important findings from the ground truth?
3. **Clinical Relevance**: Is the response clinically meaningful and appropriate?
4. **Specificity**: Does it provide appropriate level of detail (not too vague, not overly specific)?

Provide your evaluation in the following JSON format:
{{
  "accuracy_score": <1-5>,
  "completeness_score": <1-5>,
  "clinical_relevance_score": <1-5>,
  "specificity_score": <1-5>,
  "overall_score": <1-5>,
  "key_findings_matched": ["list", "of", "matched", "findings"],
  "key_findings_missed": ["list", "of", "missed", "findings"],
  "incorrect_findings": ["list", "of", "incorrect", "findings"],
  "reasoning": "Brief explanation of your evaluation"
}}

Respond ONLY with the JSON object, no other text."""

    return prompt


def evaluate_with_anthropic(client, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> Dict:
    """Evaluate using Anthropic's Claude."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        # Try to parse JSON
        # Sometimes the model wraps it in markdown, so clean it
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return {
            "accuracy_score": 0,
            "completeness_score": 0,
            "clinical_relevance_score": 0,
            "specificity_score": 0,
            "overall_score": 0,
            "key_findings_matched": [],
            "key_findings_missed": [],
            "incorrect_findings": [],
            "reasoning": f"Evaluation failed: {str(e)}"
        }


def evaluate_with_openai(client, prompt: str, model: str = "gpt-4") -> Dict:
    """Evaluate using OpenAI's GPT."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024
        )

        response_text = response.choices[0].message.content.strip()

        # Clean markdown if present
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return {
            "accuracy_score": 0,
            "completeness_score": 0,
            "clinical_relevance_score": 0,
            "specificity_score": 0,
            "overall_score": 0,
            "key_findings_matched": [],
            "key_findings_missed": [],
            "incorrect_findings": [],
            "reasoning": f"Evaluation failed: {str(e)}"
        }


def run_evaluation(
    input_file: str,
    output_file: str,
    n_samples: int = 200,
    use_anthropic: bool = True,
    api_key: str = None,
    delay: float = 1.0
):
    """Run LLM-as-a-judge evaluation on sampled examples."""

    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Sample examples
    print(f"Sampling {n_samples} examples...")
    sampled_examples = sample_examples(data, n_samples)

    # Initialize LLM client
    if use_anthropic and USE_ANTHROPIC:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or provided")
        client = anthropic.Anthropic(api_key=api_key)
        evaluate_fn = lambda prompt: evaluate_with_anthropic(client, prompt)
        print("Using Anthropic Claude for evaluation")
    elif USE_OPENAI:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or provided")
        client = openai.OpenAI(api_key=api_key)
        evaluate_fn = lambda prompt: evaluate_with_openai(client, prompt)
        print("Using OpenAI GPT for evaluation")
    else:
        raise RuntimeError("No LLM client available. Install anthropic or openai package.")

    # Run evaluation
    print(f"\nEvaluating {len(sampled_examples)} examples...")
    results = []

    for i, example in enumerate(tqdm(sampled_examples)):
        # Create judge prompt
        prompt = create_judge_prompt(
            example['Question'],
            example['Ground truth'],
            example['Generation']
        )

        # Get evaluation
        eval_result = evaluate_fn(prompt)

        # Combine with original data
        result = {
            'entry_id': example['entry_id'],
            'waveform_name': example['waveform_name'],
            'question': example['Question'],
            'ground_truth': example['Ground truth'],
            'generation': example['Generation'],
            'evaluation': eval_result
        }

        results.append(result)

        # Rate limiting
        if i < len(sampled_examples) - 1:
            time.sleep(delay)

    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    summary = calculate_summary(results)

    # Save results
    output_data = {
        'metadata': {
            'total_evaluated': len(results),
            'input_file': input_file,
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'summary': summary,
        'detailed_results': results
    }

    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Save summary separately
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'metadata': output_data['metadata'],
            'summary': summary
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Examples Evaluated: {summary['total_examples']}")
    print(f"\nAverage Scores (1-5 scale):")
    print(f"  Accuracy:           {summary['avg_accuracy']:.2f}")
    print(f"  Completeness:       {summary['avg_completeness']:.2f}")
    print(f"  Clinical Relevance: {summary['avg_clinical_relevance']:.2f}")
    print(f"  Specificity:        {summary['avg_specificity']:.2f}")
    print(f"  Overall:            {summary['avg_overall']:.2f}")
    print(f"\nScore Distribution:")
    for score in [1, 2, 3, 4, 5]:
        count = summary['overall_score_distribution'].get(str(score), 0)
        pct = (count / summary['total_examples'] * 100) if summary['total_examples'] > 0 else 0
        print(f"  Score {score}: {count:3d} ({pct:5.1f}%)")
    print(f"{'='*80}")

    return output_data


def calculate_summary(results: List[Dict]) -> Dict:
    """Calculate summary statistics from evaluation results."""

    total = len(results)

    if total == 0:
        return {}

    # Calculate averages
    avg_accuracy = sum(r['evaluation']['accuracy_score'] for r in results) / total
    avg_completeness = sum(r['evaluation']['completeness_score'] for r in results) / total
    avg_clinical = sum(r['evaluation']['clinical_relevance_score'] for r in results) / total
    avg_specificity = sum(r['evaluation']['specificity_score'] for r in results) / total
    avg_overall = sum(r['evaluation']['overall_score'] for r in results) / total

    # Score distribution
    overall_dist = {}
    for score in [1, 2, 3, 4, 5]:
        overall_dist[str(score)] = sum(1 for r in results if r['evaluation']['overall_score'] == score)

    # Count matched, missed, incorrect findings
    total_matched = sum(len(r['evaluation']['key_findings_matched']) for r in results)
    total_missed = sum(len(r['evaluation']['key_findings_missed']) for r in results)
    total_incorrect = sum(len(r['evaluation']['incorrect_findings']) for r in results)

    avg_matched = total_matched / total
    avg_missed = total_missed / total
    avg_incorrect = total_incorrect / total

    return {
        'total_examples': total,
        'avg_accuracy': avg_accuracy,
        'avg_completeness': avg_completeness,
        'avg_clinical_relevance': avg_clinical,
        'avg_specificity': avg_specificity,
        'avg_overall': avg_overall,
        'overall_score_distribution': overall_dist,
        'avg_findings_matched': avg_matched,
        'avg_findings_missed': avg_missed,
        'avg_findings_incorrect': avg_incorrect
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation for ECG generations")
    parser.add_argument("--input", required=True, help="Input JSON file with generations")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--use_anthropic", action="store_true", default=True, help="Use Anthropic Claude")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI GPT")
    parser.add_argument("--api_key", type=str, default=None, help="API key (or set via env var)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")

    args = parser.parse_args()

    # Choose provider
    use_anthropic = args.use_anthropic and not args.use_openai

    run_evaluation(
        input_file=args.input,
        output_file=args.output,
        n_samples=args.n_samples,
        use_anthropic=use_anthropic,
        api_key=args.api_key,
        delay=args.delay
    )
