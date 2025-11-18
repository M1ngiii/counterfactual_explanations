from typing import List
from pathlib import Path
import json
import re
import torch
import pandas as pd

from src.eval.load_model import load_countereval_model


METRIC_KEYS = [
    "feasibility",
    "consistency",
    "completeness",
    "trust",
    "understandability",
    "fairness",
    "complexity",
    "overall_satisfaction",
]


class CounterEvalLLMEvaluator:
    def __init__(self, max_new_tokens: int = 32, batch_size: int = 2):
        self.model, self.tokenizer = load_countereval_model()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def _build_prompt(self, cf_text: str) -> str:
        return f"""
You are evaluating the quality of a counterfactual explanation intended to change a model's prediction.

Below is a counterfactual instance produced by an algorithm:

Counterfactual:
{cf_text}

Definitions of evaluation metrics:
- Feasibility: How realistic and achievable the counterfactual is in the real world.
- Consistency: Whether the counterfactual is consistent with domain knowledge and the data distribution.
- Completeness: Whether the counterfactual provides enough changes to justify flipping the model's prediction.
- Trust: Whether the counterfactual makes the model's decision more trustworthy.
- Understandability: How easy it is to understand the counterfactual.
- Fairness: Whether the counterfactual avoids unfair or sensitive changes.
- Complexity: How simple the counterfactual is. (Rate from -2 to 2.)
- Overall Satisfaction: Overall evaluation of the counterfactual.

Respond with ONLY a single JSON object and nothing else.

The JSON must have exactly the following keys and integer scores in the given ranges:

{{
  "feasibility": 1-6,
  "consistency": 1-6,
  "completeness": 1-6,
  "trust": 1-6,
  "understandability": 1-6,
  "fairness": 1-6,
  "complexity": -2 to 2,
  "overall_satisfaction": 1-6
}}
""".strip()


    def _call_model(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None and "llama" in self.model.config.model_type:
            eos_token_id = self.tokenizer.convert_tokens_to_ids("</s>")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens * 2,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )

        gen_only = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        return decoded

    def _parse_output(self, text: str) -> dict:
        # Find the last {...} block in the text
        match = None
        for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
            match = m

        if match is None:
            data = {}
        else:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
            except Exception:
                data = {}

        # Ensure all keys exist, fill missing with None
        result = {k: data.get(k, None) for k in METRIC_KEYS}
        return result

    def evaluate(
        self,
        df: pd.DataFrame,
        cf_col: str = "cf_text",
    ) -> pd.DataFrame:

        prompts = [self._build_prompt(c) for c in df[cf_col]]

        all_results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            raw = self._call_model(batch)
            parsed = [self._parse_output(t) for t in raw]
            all_results.extend(parsed)

        scores_df = pd.DataFrame(all_results)
        return pd.concat([df.reset_index(drop=True), scores_df], axis=1)