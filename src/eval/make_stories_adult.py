import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

CF_RAW_DIR = Path("data") / "cf_raw"
CF_STORIES_DIR = Path("data") / "cf_stories"

# Normalization helpers

EDUCATION_MAP = {
    "HS-grad": "a high-school education",
    "Some-college": "some college education",
    "Bachelors": "a Bachelor's degree",
    "Masters": "a Master's degree",
    "Prof-school": "a professional school degree",
    "Assoc-voc": "an associate vocational degree",
    "Assoc-acdm": "an associate academic degree",
    "7th-8th": "7th-8th grade education",
    "11th": "11th grade education",
    "10th": "10th grade education",
}

WORKCLASS_MAP = {
    "Private": "the private sector",
    "Self-emp-not-inc": "self-employment (not incorporated)",
    "Self-emp-inc": "self-employment (incorporated)",
    "Local-gov": "local government",
    "State-gov": "state government",
    "Federal-gov": "federal government",
    "Without-pay": "an unpaid position",
    "Never-worked": "no prior work experience",
    "?": "an unspecified work sector",
}

RELATIONSHIP_MAP = {
    "Husband": "married and living with your spouse",
    "Wife": "married and living with your spouse",
    "Not-in-family": "not living in a family household",
    "Own-child": "living with your own child",
    "Unmarried": "not currently married",
    "Other-relative": "living with other relatives",
}

OCCUPATION_MAP = {
    "Prof-specialty": "a professional specialist",
    "Craft-repair": "a craft and repair worker",
    "Exec-managerial": "an executive or manager",
    "Adm-clerical": "an administrative clerk",
    "Sales": "a sales worker",
    "Other-service": "a service worker",
    "Machine-op-inspct": "a machine operator or inspector",
    "Transport-moving": "a transport or moving worker",
    "Handlers-cleaners": "a handler or cleaner",
    "Farming-fishing": "a farming or fishing worker",
    "Tech-support": "a technical support worker",
    "Protective-serv": "a protective service worker",
    "Priv-house-serv": "a private household service worker",
    "Armed-Forces": "a member of the armed forces",
    "?": "an unspecified occupation",
}

RACE_MAP = {
    "White": "White",
    "Black": "Black",
    "Asian-Pac-Islander": "Asian-Pac-Islander",
    "Amer-Indian-Eskimo": "American Indian or Eskimo",
    "Other": "unspecified race",
    "?": "unspecified race",
}

# Helpers
def parse_kv_text(text: str) -> Dict[str, str]:
    """ Parse csv into dictionary """
    parts = [p.strip() for p in text.split(",")]
    kv = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv

def normalize_education(raw: str) -> str:
    if not raw:
        return ""
    if raw in EDUCATION_MAP:
        return EDUCATION_MAP[raw]
    return raw.replace("-", " ") + " education"

def normalize_workclass(raw: str) -> str:
    if not raw:
        return ""
    return WORKCLASS_MAP.get(raw, raw.replace("-", " "))

def normalize_relationship(raw: str) -> str:
    if not raw:
        return ""
    return RELATIONSHIP_MAP.get(raw, raw.replace("-", " ").lower())

def normalize_occupation(raw: str) -> str:
    if not raw:
        return ""
    return OCCUPATION_MAP.get(raw, raw.replace("-", " ").lower())

def normalize_country(raw: str) -> str:
    if not raw or raw == "?":
        return "an unspecified country"
    return raw.replace("-", " ")

def normalize_race(raw: str) -> str:
    if not raw:
        return ""
    return RACE_MAP.get(raw, raw.replace("-", " "))

def adult_factual_story(feat: Dict[str, str]) -> str:
    """Turn a parsed Adult row into a short natural-language scenario."""

    age = feat.get("age", "unknown")
    sex = feat.get("sex", "")  # "Male" / "Female"
    marital_raw = feat.get("marital-status", "")
    education_raw = feat.get("education", "")
    hours = feat.get("hours-per-week", "unknown")
    occupation_raw = feat.get("occupation", "")
    workclass_raw = feat.get("workclass", "")
    relationship_raw = feat.get("relationship", "")
    country_raw = feat.get("native-country", "")
    race_raw = feat.get("race", "")

    marital = marital_raw.replace("-", " ").lower()
    country_phrase = normalize_country(country_raw)
    occupation_phrase = normalize_occupation(occupation_raw)
    workclass_phrase = normalize_workclass(workclass_raw)
    education_phrase = normalize_education(education_raw)
    relationship_phrase = normalize_relationship(relationship_raw)
    race_phrase = normalize_race(race_raw)

    parts = []

    # basic identity
    if sex:
        if race_phrase:
            parts.append(f"You are a {age}-year-old {race_phrase} {marital} {sex}.")
        else:
            parts.append(f"You are a {age}-year-old {marital} {sex}.")
    else:
        if race_phrase:
            parts.append(f"You are a {age}-year-old {race_phrase} and {marital}.")
        else:
            parts.append(f"You are {age} years old and {marital}.")

    # education
    if education_phrase:
        parts.append(f"You have {education_phrase}.")

    # work
    if hours and occupation_phrase:
        if workclass_phrase:
            parts.append(
                f"You work {hours} hours per week as {occupation_phrase} in {workclass_phrase}."
            )
        else:
            parts.append(f"You work {hours} hours per week as {occupation_phrase}.")
    elif hours:
        parts.append(f"You work {hours} hours per week.")

    # relationship / household
    if relationship_phrase:
        parts.append(f"You are {relationship_phrase}.")
    elif relationship_raw:
        parts.append(f"You are {relationship_raw.replace('-', ' ').lower()}.")

    # location
    parts.append(f"You live in {country_phrase}.")

    return " ".join(p.strip() for p in parts if p)

def adult_change_story(orig: Dict[str, str], cf: Dict[str, str]) -> str:
    changes = []

    for key in orig.keys():
        if key not in cf or orig[key] == cf[key]:
            continue

        nice_name = key.replace("-", " ")

        if key == "education":
            from_val = normalize_education(orig[key])
            to_val = normalize_education(cf[key])
        elif key == "workclass":
            from_val = normalize_workclass(orig[key])
            to_val = normalize_workclass(cf[key])
        elif key == "relationship":
            from_val = normalize_relationship(orig[key])
            to_val = normalize_relationship(cf[key])
        elif key == "native-country":
            from_val = normalize_country(orig[key])
            to_val = normalize_country(cf[key])
        elif key == "occupation":
            from_val = normalize_occupation(orig[key])
            to_val = normalize_occupation(cf[key])
        elif key == "race":
            from_val = normalize_race(orig[key])
            to_val = normalize_race(cf[key])
        else:
            from_val = orig[key].replace("-", " ")
            to_val = cf[key].replace("-", " ")

        changes.append(f"Change {nice_name} from {from_val} to {to_val}")

    if not changes:
        return "No changes are required; the situation already leads to the desired outcome."

    return "; ".join(changes) + "."


def build_stories_for_row(orig_text: str, cf_text: str) -> str:
    """ Take the original orig_text/cf_text key=value strings and return one full story prompt """

    orig_feat = parse_kv_text(orig_text)
    cf_feat = parse_kv_text(cf_text)

    factual_story = adult_factual_story(orig_feat)
    change_story = adult_change_story(orig_feat, cf_feat)

    # Single narrative assuming we go from low income -> higher than average income
    full_story = f"""Imagine you are in the following scenario: {factual_story} Current outcome: You are earning less than the average salary. Useful context: The standard full-time workload is 40 hours per week. To earn more than the average salary, you would need to make the following changes: {change_story} The rest of the values will remain constant."""
    return full_story


# Main pipeline

def make_stories_for_algorithm(algorithm: str):
    algorithm = algorithm.upper()
    in_path = CF_RAW_DIR / f"{algorithm.lower()}.csv"

    if not in_path.exists():
        raise FileNotFoundError(f"Could not find input file: {in_path}")

    print(f"Loading counterfactuals from {in_path}...")
    df = pd.read_csv(in_path)

    if "orig_text" not in df.columns or "cf_text" not in df.columns:
        raise ValueError("Input CSV must contain 'orig_text' and 'cf_text' columns")

    # Build stories row by row
    stories = []

    for _, row in df.iterrows():
        story = build_stories_for_row(row["orig_text"], row["cf_text"])
        stories.append(story)

    df_stories = df.copy()
    df_stories["cf_story"] = stories

    # Drop the old raw text columns
    df_stories = df_stories.drop(columns=["orig_text", "cf_text"])

    # Ensure story is the first column
    cols = ["cf_story"] + [c for c in df_stories.columns if c != "cf_story"]
    df_stories = df_stories[cols]

    CF_STORIES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CF_STORIES_DIR / f"{algorithm.lower()}_stories.csv"
    df_stories.to_csv(out_path, index=False)

    print(f"Saved story-based counterfactuals to {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        default="FACE",
        help="Which algorithm to convert (DiCE, GS, AR, FACE, CLUE, or All)",
    )

    args = parser.parse_args()

    algorithms = ["DiCE", "GS", "AR", "FACE", "CLUE"]

    if args.algorithm.lower() == "all":
        for algo in algorithms:
            try:
                print(f"\nConverting {algo}")
                make_stories_for_algorithm(algo)
            except FileNotFoundError as e:
                print(f"Skipping {algo}: {e}")
    else:
        make_stories_for_algorithm(args.algorithm)


if __name__ == "__main__":
    main()
