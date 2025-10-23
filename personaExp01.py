import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any
import random
import pandas as pd
from tqdm import tqdm
import ollama



def build_persona_description(row: pd.Series) -> str:
    return (
        f"- PersonaID: {row.get('PersonaID')}\n"
        f"- AgeGroup: {row.get('AgeGroup')}\n"
        f"- Gender: {row.get('Gender')}\n"
        f"- EducationLevel: {row.get('EducationLevel')}\n"
        f"- OccupationSector: {row.get('OccupationSector')}\n"
        f"- Region: {row.get('Region')}\n"
        f"- PoliticalIdeology: {row.get('PoliticalIdeology')}\n"
        f"- Trust_ScienceInstitutions: {row.get('Trust_ScienceInstitutions')}\n"
        f"- Belief_ClimateExists: {row.get('Belief_ClimateExists')}\n"
        f"- Belief_HumanContribution: {row.get('Belief_HumanContribution')}\n"
        f"- Emotional_WorryAboutClimate: {row.get('Emotional_WorryAboutClimate')}\n"
        f"- BehaviouralOrientation: {row.get('BehaviouralOrientation')}\n"
        f"- SocialConnectivity: {row.get('SocialConnectivity')}"
    )


def chat_once(model: str, temperature: float, system_msg: str, user_msg: str) -> str:
    r = ollama.chat(
        model=model,
        options={"temperature": temperature},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return r["message"]["content"].strip()


def coerce_json(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1 or e <= s:
            raise ValueError(f"No JSON object found in: {text[:120]}...")
        obj = json.loads(text[s:e + 1])

    claim = str(obj.get("claimStance", "")).strip()
    belief = str(obj.get("climateChangeStance", obj.get("climateChnageStance", ""))).strip()

    CLAIM_ALLOWED = {"Support", "Not Support"}
    BELIEF_ALLOWED = {
        "Strongly disagree", "Slightly Disagree", "Neutral", "Slightly Agree", "Strongly Agree"
    }

    if claim not in CLAIM_ALLOWED:
        claim = "Support" if "support" in claim.lower() and "not" not in claim.lower() else "Not Support"
    canon_map = {v.lower(): v for v in BELIEF_ALLOWED}
    belief = canon_map.get(belief.lower(), "Neutral")

    return {"claimStance": claim, "climateChangeStance": belief}


def load_claims_with_label_JSON(path: str):

    raw = json.loads(Path(path).read_text())
    out = []
    for it in raw:
        claim_text = (
            it.get("claim") or it.get("claim_text") or it.get("statement") or it.get("text") or ""
        )
        if not claim_text:
            continue
        out.append({
            "claim_id": it.get("claim_id") or it.get("id"),
            "claim_text": claim_text,
            "claim_stance_label": it.get("label") or it.get("claim_label") or it.get("verdict") or it.get("stance")
        })
    return out

def load_claims_with_label_CSV(path: str):
    try:
        df = pd.read_csv(Path(path))
    except Exception as e:
        print(f"Error reading CSV file at {path}: {e}")
        return []

    raw = df.to_dict('records')
    out = []
    for it in raw:
        claim_text = (
            it.get("claim") or it.get("claim_text") or it.get("statement") or it.get("text") or ""
        )
        if not claim_text:
            continue
        
        out.append({
            "claim_id": it.get("claim_id") or it.get("id"),
            "claim_text": claim_text,
            "claim_stance_label": it.get("stance_label")
        })
        
    return out

def filter_balanced_claims(claims: list, n_each: int = 100) -> list:
   
    df = pd.DataFrame(claims)
    if "claim_stance_label" not in df.columns:
        raise ValueError("Expected 'claim_stance_label' field in claims.")

    df["claim_stance_label"] = df["claim_stance_label"].astype(str).str.upper()

    supports = df[df["claim_stance_label"] == "SUPPORTS"]
    refutes = df[df["claim_stance_label"] == "REFUTES"]

    n_each = min(n_each, len(supports), len(refutes))
    supports_sample = supports.sample(n=n_each, random_state=42)
    refutes_sample = refutes.sample(n=n_each, random_state=42)

    balanced = pd.concat([supports_sample, refutes_sample]).sample(frac=1, random_state=42).to_dict(orient="records")
    return balanced



def main():
  
        
    parser = argparse.ArgumentParser(
        description="Simulate persona-based climate claim evaluations using an LLM model via Ollama."
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--personas", required=True, help="Path to personas CSV file")
    parser.add_argument("--claims", required=True, help="Path to claims CSV or JSON file")
    parser.add_argument("--out", default="agents_claim_outputs.json", help="Output JSON filename")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--n_personas", type=int, default=None, help="Number of personas to use (optional)")
    parser.add_argument("--n_claims", type=int, default=200, help="Number of claims to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (set None to disable)")

    args = parser.parse_args()
    personas = pd.read_csv(args.personas)
    claims = load_claims_with_label_CSV(args.claims)
    claims = filter_balanced_claims(claims, n_each=100)
    if args.n_personas is None:
        args.n_personas = len(personas)

    print(f"Loaded {len(personas)} personas and {len(claims)} filtered claims.")   

    

    if args.seed is not None:
        personas = personas.sample(n=min(args.n_personas, len(personas)), random_state=args.seed)
        random.seed(args.seed)
        claims = (
            pd.DataFrame(claims)
            .sample(n=min(args.n_claims, len(claims)), random_state=args.seed)
            .to_dict(orient="records")
        )
    else:
        personas = personas.head(args.n_personas)
        claims = claims[:args.n_claims]

    SYSTEM_TMPL = """You are simulating the thought process of a person with the following persona:

{PERSONA_DESCRIPTION}

Your task is to evaluate claims about climate change.
- Always respond as this person would, considering their background, values, and beliefs.
- You cannot access new facts beyond what is given.
- Your reasoning may be influenced by your prior beliefs (this is natural).
- Be consistent in personality and tone across all answers.

For each claim you see, you will:
1. Read the claim carefully.
2. Decide whether you accept the claim or not.
3. Give your stance on whether you support or not support the claim. You should respond by either "Support" or "Not Support".
4. Give your stance on climate change existence. Respond with: "Strongly disagree", "Slightly Disagree", "Neutral", "Slightly Agree", or "Strongly Agree".
"""

    CLAIM_TMPL = """Claim: {CLAIM_TEXT}
Given the above claim, return:
{{
"climateChnageStance": "...",
"claimStance": "..."
}}"""

    records = []
    for _, prow in tqdm(personas.iterrows(), total=len(personas), desc="Personas"):
        persona_desc = build_persona_description(prow)
        system_msg = SYSTEM_TMPL.replace("{PERSONA_DESCRIPTION}", persona_desc)

        for c in claims:
            user_msg = CLAIM_TMPL.replace("{CLAIM_TEXT}", str(c["claim_text"]))
            try:
                raw = chat_once(args.model, args.temperature, system_msg, user_msg)
                parsed = coerce_json(raw)
            except Exception:
                parsed = {"claimStance": "Not Support", "climateChangeStance": "Neutral"}
                raw = "Parsing failed"

            records.append({
                "persona_id": prow.get("PersonaID"),
                "belief_climate_exists": prow.get("Belief_ClimateExists"),
                "claim_id": c["claim_id"],
                "claim": c["claim_text"],
                "claim_stance_label": c.get("claim_stance_label"),
                "llm_responses": parsed,
                "raw": raw
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()