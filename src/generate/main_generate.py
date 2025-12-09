import argparse
import importlib
from pathlib import Path
import pandas as pd
import os
import warnings

from src.carla_wrappers.datasets.adult import AdultData
from src.carla_wrappers.models.ann_adult import AdultANNModel
from src.generate.adult_model import load_adult_model 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


ALGORITHM_REGISTRY = {
    "DiCE_legacy": "DiCE.generator:DiCEGenerator",
    "Carla": "Carla.generator:CarlaGenerator",
}

CF_RAW_DIR = Path("data") / "cf_raw"


def load_generator(name: str):
    if name in {"GS", "AR", "FACE", "CLUE", "DiCE"}:
        name = "Carla"
    module_path, class_name = ALGORITHM_REGISTRY[name].split(":")
    module = importlib.import_module(f"src.generate.algorithms.{module_path}")
    cls = getattr(module, class_name)
    return cls



def run_algorithm(algorithm: str, n_factuals: int):
    GeneratorClass = load_generator(algorithm)

    dataset = AdultData()
    dataset_name = "adult"
    model = AdultANNModel(dataset)
    model_name = "ann"

    OVERSAMPLE_FACTORS = {
        "GS": 10,
        "AR": 10,
        "CLUE": 25,
        "DiCE": 2,
        "FACE": 1,
    }

    if algorithm in {"GS", "AR", "FACE", "CLUE", "DiCE"}: 
        oversample_factor = OVERSAMPLE_FACTORS.get(algorithm, 1)
        generator = GeneratorClass(dataset=dataset, dataset_name=dataset_name, model=model, model_name=model_name, method=algorithm, oversample=oversample_factor)

    elif algorithm == "DiCE_legacy":
        bundle = load_adult_model()
        generator = GeneratorClass(bundle=bundle)

    else:
        print(f"Unknown algorithm: {algorithm}")
        return

    print(f"Generating counterfactuals using {algorithm}...")
    df_cf: pd.DataFrame = generator.generate(n_factuals=n_factuals)

    CF_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CF_RAW_DIR / f"{algorithm.lower()}.csv"
    df_cf.to_csv(out_path, index=False)

    print(f"Saved raw counterfactuals for {algorithm} to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        default="FACE",
        help="Which recourse algorithm to run (DiCE, GS, AR, FACE, CLUE or All)",
    )
    parser.add_argument(
        "--n_factuals",
        type=int,
        default=50,
        help="Number of factual instances to generate CFs for",
    )
    args = parser.parse_args()
    
    SUPPORTED_ALGS = ["GS", "AR", "FACE", "CLUE", "DiCE"]

    if args.algorithm == "All":
        algorithms_to_run = SUPPORTED_ALGS
    else:
        algorithms_to_run = [args.algorithm]

    # Run the algs
    for algo in algorithms_to_run:
        run_algorithm(algo, args.n_factuals)


if __name__ == "__main__":
    main()