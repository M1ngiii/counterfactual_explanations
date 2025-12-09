import argparse
from pathlib import Path

import pandas as pd

from src.eval.countereval_llm import CounterEvalLLMEvaluator

ALGORITHMS = ["DiCE", "GS", "AR", "FACE", "CLUE"]            # ["DiCE", "GS", "AR", "FACE", "CLUE"]

CF_STORY_DIR = Path("data") / "cf_stories"
CF_SCORED_DIR = Path("data") / "cf_scored"


def run_evaluation(algorithm: str):
    """Load storified CFs for an algorithm, score them with CounterEval, save to CSV."""

    in_path = CF_STORY_DIR / f"{algorithm.lower()}_stories.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find input file: {in_path}")

    print(f"Loading counterfactuals from {in_path}...")
    df = pd.read_csv(in_path)

    # Sanity check: make sure column exists
    if "cf_story" not in df.columns:
        raise ValueError("Input CSV must contain a 'cf_story' column")

    print("Loading CounterEval LLM evaluator...")
    evaluator = CounterEvalLLMEvaluator(
        max_new_tokens=64,
        batch_size=2,
    )

    print(f"Scoring {len(df)} counterfactuals for {algorithm}...")
    scored_df = evaluator.evaluate(df, cf_col="cf_story")

    CF_SCORED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CF_SCORED_DIR / f"{algorithm.lower()}_scored.csv"
    scored_df.to_csv(out_path, index=False)

    print(f"Saved scored counterfactuals to {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        default="DiCE",
        help="Which algorithm to evaluate (DiCE, GS, AR, FACE, CLUE, or All)",
    )

    args = parser.parse_args()

    if args.algorithm.lower() == "all":
        for algo in ALGORITHMS:
            try:
                run_evaluation(algo)
            except FileNotFoundError as e:
                print(f"Skipping {algo}: {e}")
    else:
        run_evaluation(args.algorithm)


if __name__ == "__main__":
    main()