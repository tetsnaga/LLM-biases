#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import random
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm
import ollama


# ---------------------------
# Likert utilities
# ---------------------------
LIKERT_ORDER = [
    "Strongly disagree", "Slightly Disagree", "Neutral",
    "Slightly Agree", "Strongly Agree"
]
LIKERT_TO_NUM = {v: i-2 for i, v in enumerate(LIKERT_ORDER, start=0)}  # {-2..+2}


# ---------------------------
# Prompt templates
# ---------------------------
DEFAULT_PERSONA_TMPL = (
    "- PersonaID: {PersonaID}\n"
    "- AgeGroup: {AgeGroup}\n"
    "- Gender: {Gender}\n"
    "- EducationLevel: {EducationLevel}\n"
    "- OccupationSector: {OccupationSector}\n"
    "- Region: {Region}\n"
    "- PoliticalIdeology: {PoliticalIdeology}\n"
    "- Trust_ScienceInstitutions: {Trust_ScienceInstitutions}\n"
    "- Belief_ClimateExists: {Belief_ClimateExists}\n"
    "- Belief_HumanContribution: {Belief_HumanContribution}\n"
    "- Emotional_WorryAboutClimate: {Emotional_WorryAboutClimate}\n"
    "- BehaviouralOrientation: {BehaviouralOrientation}\n"
    "- SocialConnectivity: {SocialConnectivity}"
)

DEFAULT_SYSTEM_TMPL = """You are simulating the thought process of a person with the following persona:

{PERSONA_DESCRIPTION}

Let me explain what each part of your persona means:
    - Your PersonaID is just a unique label so I can tell you apart from others.
    - Your AgeGroup shows the general age range you belong to, like 18–24 or 25–34.
    - Your Gender is how you identify yourself — male, female, or non-binary.
    - Your EducationLevel tells your level of education.
    - Your OccupationSector describes the kind of work or industry you’re in.
    - Your Region is where you live in the world — for example, the US South or the Global South — which shapes your local experiences.
    - Your PoliticalIdeology shows where you generally stand on the liberal-to-conservative spectrum.
    - Your Trust_ScienceInstitutions shows how much you trust science and research organizations.
    - Your Belief_ClimateExists tells how strongly you believe that climate change is real.
    - Your Emotional_WorryAboutClimate reflects how personally worried or emotionally affected you feel about climate change.
    - Your BehaviouralOrientation shows how motivated you are to take or support climate-positive actions.
    - Your SocialConnectivity describes how socially active and connected you are in your community or networks.

Your task is to evaluate claims about climate change.
- Each claim has a certatin validity level that will be given to you with the claim that you can use to make your decision.
- Always respond as this person would, considering their background, values, and beliefs.
- You cannot access new facts beyond what is given.
- Your reasoning may be influenced by your prior beliefs (this is natural).
- Be consistent in personality and tone across all answers.

For each claim you see, you will:
1. Read the claim carefully.
2. Decide whether you accept the claim or not.
3. Give your stance on whether you support or not support the claim or you are neutral. You should respond by either "Accept" or "Refute" or "Neutral".
4. Give your reasoning for your stance in item 3.
5. Give your stance on climate change existence. Respond with: "Strongly disagree", "Slightly Disagree", "Neutral", "Slightly Agree", or "Strongly Agree".
6. Give your reasoning for your stance in item 5.
"""

DEFAULT_CLAIM_TMPL = """Claim: {CLAIM_TEXT}

Evidence Strength: {EVIDENCE_LEVEL}

Given the above claim and evidence, return ONLY a single-line JSON object (no extra text, no code fences):
you should provide your stance on the claim whether you accept it, refute it or are neutral about it (claim_decision). then you should provide your reasoning for your (claim_decision_reason).
Also provide your stance toward climate change existance between Strongly disagree or Slightly Disagree or Neutral or Slightly Agree or Strongly Agree (climateChange_belief). Then provide your reason for your climate change existance stance (climateChange_belief_reason). 
Your final answer should be in the following format:
{
"claim_decision":"<Accept | Neutral | Refute>",
"claim_decision_reason":"your reason",
"climateChange_belief":"<Strongly disagree | Slightly Disagree | Neutral | Slightly Agree | Strongly Agree>",
"climateChange_belief_reason":"your reason",
}

make sure to following exact spellings.
"""


# ---------------------------
# Utility helpers
# ---------------------------
def render_template(row: pd.Series, template: str) -> str:
    class SafeDict(dict):
        def __missing__(self, key): return ""
    return template.format_map(SafeDict(**{k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}))


def ensure_model_available(model: str) -> None:
    try:
        if hasattr(ollama, "show"):
            ollama.show(model=model)
        else:
            models = ollama.list().get("models", [])
            names = {m.get("name") for m in models}
            if model not in names:
                raise RuntimeError(f"Model '{model}' not found.")
    except Exception as ex:
        raise SystemExit(f"Ollama model '{model}' not available. Run:  ollama pull {model}\nError: {ex}")


def chat_once(model: str, temperature: float, system_msg: str, user_msg: str, top_p: Optional[float]=None) -> str:
    options: Dict[str, Any] = {"temperature": temperature}
    if top_p is not None:
        options["top_p"] = top_p
    r = ollama.chat(
        model=model,
        options=options,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return r["message"]["content"].strip()

def coerce_json_extended(text: str) -> Dict[str, Any]:
    # expected output skeleton
    out = {
        "claim_decision": "",
        "claim_decision_reason": "",
        "climateChange_belief": "",
        "climateChange_belief_reason": "",
    }

    # --- 1) Extract the JSON-ish substring (handles fenced code blocks) ---
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").replace("```json", "").replace("```", "").strip()
    s, e = t.find("{"), t.rfind("}")
    candidate = t[s:e+1] if (s != -1 and e != -1 and e > s) else t

    # --- 2) Try JSON parse with key normalization (handles snake_case/camelCase) ---
    def norm(k: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (k or "").lower())

    canonical = {
        "claimdecision": "claim_decision",
        "decision": "claim_decision",
        "claimdecisionreason": "claim_decision_reason",
        "decisionreason": "claim_decision_reason",
        "climatechangebelief": "climateChange_belief",
        "climatebelief": "climateChange_belief",
        "climatechangebeliefreason": "climateChange_belief_reason",
        "climatebeliefreason": "climateChange_belief_reason",
    }

    try:
        obj = json.loads(candidate)
        for k, v in (obj if isinstance(obj, dict) else {} ).items():
            ck = canonical.get(norm(k))
            if ck in out:
                out[ck] = ("" if v is None else str(v)).strip()
    except Exception:
        # --- 3) Regex fallback: accept multiple key variants ---
        patterns = {
            "claim_decision": r'"(?:claim_decision|claimDecision|decision)"\s*:\s*"([^"]+)"',
            "claim_decision_reason": r'"(?:claim_decision_reason|claimDecisionReason|decision_reason|decisionReason)"\s*:\s*"([^"]+)"',
            "climateChange_belief": r'"(?:climateChange_belief|climateChangeBelief|climate_belief|climateBelief)"\s*:\s*"([^"]+)"',
            "climateChange_belief_reason": r'"(?:climateChange_belief_reason|climateChangeBeliefReason|climate_belief_reason|climateBeliefReason)"\s*:\s*"([^"]+)"',
        }
        for key, pat in patterns.items():
            m = re.search(pat, candidate, re.I)
            if m:
                out[key] = m.group(1).strip()

    # --- 4) Post-process / normalize values ---
    if out["claim_decision"] not in {"Accept", "Neutral", "Refute"}:
        txt = out["claim_decision"].lower()
        if any(w in txt for w in ("refute", "reject", "not support", "oppose", "deny")):
            out["claim_decision"] = "Refute"
        elif any(w in txt for w in ("accept", "support", "agree", "endorse")):
            out["claim_decision"] = "Accept"
        else:
            out["claim_decision"] = "Neutral"

    # default climate belief if unknown (assumes LIKERT_TO_NUM exists in caller)
    try:
        if out["climateChange_belief"] not in LIKERT_TO_NUM:
            out["climateChange_belief"] = "Neutral"
    except NameError:
        # if mapping isn't available here, still coerce to a sane default
        if out["climateChange_belief"] == "":
            out["climateChange_belief"] = "Neutral"

    return out

def load_claims(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
    else:
        data = pd.read_csv(p).to_dict("records")
    claims = []
    for it in data:
        txt = it.get("claim") or it.get("claim_text") or it.get("statement") or it.get("text") or ""
        if not txt:
            continue
        claims.append({
            "claim_id": it.get("claim_id") or it.get("id"),
            "claim_text": txt,
            "claim_stance_label": it.get("stance_label") or it.get("label") or it.get("verdict"),
            "evidence_level": it.get("evidence_level") or it.get("evidence_strength") or it.get("evidence"),
        })
    return claims


def to_signed_label(label: str) -> int:
    lab = (str(label) or "").upper()
    if lab == "SUPPORTS": return +1
    if lab == "REFUTES":  return -1
    return -1


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="PersonaClaim evaluation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--personas", required=False, default='Data/synthetic_climate_personas.csv')
    ap.add_argument("--claims", required=False,default='Data/climate-fever-dataset.csv')
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--n_personas", type=int, default=None)
    ap.add_argument("--n_claims", type=int, default=200)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_model_available(args.model)

    personas = pd.read_csv(args.personas)
    claims = load_claims(args.claims)
    dfc = pd.DataFrame(claims)
    if "claim_stance_label" in dfc.columns:
        dfc["t_j"] = dfc["claim_stance_label"].map(to_signed_label)
    else:
        dfc["t_j"] = 0


    if args.n_personas:
            personas = personas.sample(n=min(args.n_personas, len(personas)))
    if args.n_claims:
            dfc = dfc.sample(n=min(args.n_claims, len(dfc)))

    system_prompt_raw = DEFAULT_SYSTEM_TMPL
    claim_prompt_raw = DEFAULT_CLAIM_TMPL
    persona_tmpl = DEFAULT_PERSONA_TMPL

    records = []
    for _, prow in tqdm(personas.iterrows(), total=len(personas), desc="Personas"):
        persona_id = prow.get("PersonaID")
        persona_desc = render_template(prow, persona_tmpl)
        system_msg = system_prompt_raw.replace("{PERSONA_DESCRIPTION}", persona_desc)
        base_belief = str(prow.get("Belief_ClimateExists", "Neutral"))
        dfp = dfc.sample(frac=1, random_state=random.randint(0, 10_000))

        for _, row in dfp.iterrows():
            claim_id = row.get("claim_id")
            claim_text = str(row.get("claim_text"))
            claim_label = int(row.get("t_j", 0))
            evidence = row.get("evidence_level") or "Moderate"
            user_msg = claim_prompt_raw.replace("{CLAIM_TEXT}", claim_text).replace("{EVIDENCE_LEVEL}", str(evidence))
            raw = chat_once(args.model, args.temperature, system_msg, user_msg, top_p=args.top_p)
            print(raw)
            parsed = coerce_json_extended(raw)
            print(parsed)
            records.append({
                "PersonaID": persona_id,
                "BeliefClimateExists": base_belief,
                "ClaimID": claim_id,
                "ClaimStanceLabel": claim_label,
                "EvidenceStrength": evidence,
                "ModelDecisionOfClaim": parsed["claim_decision"],
                "ModelDecisionOfClaim_Reason": parsed["claim_decision_reason"],
                "ModelBeliefClimateExists": parsed["climateChange_belief"],
                "ModelBeliefClimateExists_Reason": parsed["climateChange_belief_reason"],
            })

    df_out = pd.DataFrame(records)
    cols = [
        "PersonaID",
        "BeliefClimateExists",
        "ClaimID",
        "ClaimStanceLabel",
        "EvidenceStrength",
        "ModelDecisionOfClaim",
        "ModelDecisionOfClaim_Reason",
        "ModelBeliefClimateExists",
        "ModelBeliefClimateExists_Reason",
    ]
    df_out = df_out[cols]
    out_path = outdir / "decisions.csv"
    df_out.to_csv(out_path, index=False)
    print(f"wrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
