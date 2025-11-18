import argparse
from pathlib import Path

import pandas as pd

from src.eval.countereval_llm import CounterEvalLLMEvaluator

CF_RAW_DIR = Path("data") / "cf_raw"
CF_SCORED_DIR = Path("data") / "cf_scored"


def run_evaluation(algorithm: str):
    """Load raw CFs for an algorithm, score them with CounterEval, save to CSV."""

    in_path = CF_RAW_DIR / f"{algorithm.lower()}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find input file: {in_path}")

    print(f"Loading counterfactuals from {in_path}...")
    df = pd.read_csv(in_path)

    # Sanity check: make sure columns exist
    if "orig_text" not in df.columns or "cf_text" not in df.columns:
        raise ValueError("Input CSV must contain 'orig_text' and 'cf_text' columns")

    print("Loading CounterEval LLM evaluator...")
    evaluator = CounterEvalLLMEvaluator(
        max_new_tokens=64,
        batch_size=2,
    )

    print(f"Scoring {len(df)} counterfactuals for {algorithm}...")
    scored_df = evaluator.evaluate(df)

    CF_SCORED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CF_SCORED_DIR / f"{algorithm.lower()}_scored.csv"
    scored_df.to_csv(out_path, index=False)

    print(f"Saved scored counterfactuals to {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        default="DiCE",
        help="Which algorithm to evaluate (DiCE, GS, AR, FACE, CLUE)",
    )

    args = parser.parse_args()

    run_evaluation(args.algorithm)


if __name__ == "__main__":
    main()