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


from sample_personas import sample_personas_dataset
from sample_claims import sample_claims_dataset

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
from prompts import *

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
        txt = it.get("claim"),
        claims.append({
            "claim_id": it.get("claim_id"),
            "claim_text": txt,
            "claim_stance_label": it.get("stance_label"),
            "claim_label": it.get("claim_label"),
            "evidences" : it.get("evidences") or [],
            "source_type": it.get("source_type") or "None",
            "claim_entropy": it.get("claim_entropy") or "None"
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
    ap.add_argument("--personas", required=True, default='Data/synthetic_climate_personas.csv')
    ap.add_argument("--claims", required=True,default='Data/climate-fever-dataset.csv')
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--n_personas", type=int, default=None)
    ap.add_argument("--n_claims", type=int, default=None)
    ap.add_argument("--evidenceLevel",type=int)
    ap.add_argument("--n_evidence", type=int, default=1)
    ap.add_argument("--evidence_random", required=False, default=False)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_model_available(args.model)

    personas = pd.read_csv(args.personas)    
    # personas = sample_personas_dataset(args.personas, sample_size=100, random_seed=42)
    # print(personas)
    # claims = load_claims(args.claims)
    claims = load_claims(args.claims)
    # print("DEBUG type(claims):", type(claims))
    # print("DEBUG claims:", claims)


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
    print(args.evidenceLevel)
    if args.evidenceLevel == 1:
        claim_prompt_raw = DEFAULT_CLAIM_TMPL_LEVEL1
    elif args.evidenceLevel == 2:
        claim_prompt_raw = DEFAULT_CLAIM_TMPL_LEVEL2
    elif args.evidenceLevel == 3:
        claim_prompt_raw = DEFAULT_CLAIM_TMPL_LEVEL3
    elif args.evidenceLevel == 4:
        claim_prompt_raw = DEFAULT_CLAIM_TMPL_LEVEL4
    else:
        raise("Enter a valid evidence level (1,2,3, or 4)")
        
        

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
            claim_stance_label = str(row.get("claim_stance_label"))
            claim_entropy = None
            ev_source = None
            ev_ids = []
            evs = []
            ev_text = ""
            
            evidencelevel = args.evidenceLevel
            if evidencelevel == 1:
                user_msg = claim_prompt_raw.replace("{CLAIM_TEXT}", claim_text)

            elif evidencelevel == 2:
                evs = pick_evidences(row, args.n_evidence, args.evidence_random)
                ev_text = join_evidence_text(evs)
                user_msg = (claim_prompt_raw
                            .replace("{CLAIM_TEXT}", claim_text)
                            .replace("{EVIDENCE}", ev_text))

            elif evidencelevel == 3:
                evs = pick_evidences(row, args.n_evidence, args.evidence_random)
                ev_text = join_evidence_text(evs) 
                ev_ids = join_evidence_ids(evs)
                src_type = row.get("source_type")
                ev_source = f"({src_type})"
                user_msg = (claim_prompt_raw
                            .replace("{CLAIM_TEXT}", claim_text)
                            .replace("{EVIDENCE}", ev_text)
                            .replace("{EVIDENCE_SOURCE}", ev_source))

            elif evidencelevel == 4:
                evs = pick_evidences(row, args.n_evidence, args.evidence_random)
                ev_text = join_evidence_text(evs)
                claim_entropy = str(row.get("claim_entropy"))
                user_msg = (claim_prompt_raw
                            .replace("{CLAIM_TEXT}", claim_text)
                            .replace("{EVIDENCE}", ev_text)
                            .replace("{CLAIM_ENTROPY}", claim_entropy))  

            else:
                raise ValueError("Enter a valid evidence level (1, 2, 3, or 4)")



            raw = chat_once(args.model, args.temperature, system_msg, user_msg, top_p=args.top_p)
            parsed = coerce_json_extended(raw)
            records.append({
                "PersonaID": persona_id,
                "BeliefClimateExists": base_belief,
                "ClaimID": claim_id,
                "ClaimStanceLabel": claim_stance_label,
                "EvidencesVerdict": row.get("claim_label"),
                "EvidenceLevel": evidencelevel,
                "ModelDecisionOfClaim": parsed["claim_decision"],
                "ModelDecisionOfClaim_Reason": parsed["claim_decision_reason"],
                "ModelBeliefClimateExists": parsed["climateChange_belief"],
                "ModelBeliefClimateExists_Reason": parsed["climateChange_belief_reason"],
                "Evidence_ids": ev_ids,
                "Evidence_source": ev_source,
                "Evidence": ev_text,
                "ClaimEntropy": claim_entropy


            })

    df_out = pd.DataFrame(records)
    cols = [
        "PersonaID",
        "BeliefClimateExists",
        "ClaimID",
        "ClaimStanceLabel",
        "EvidenceLevel",
        "ModelDecisionOfClaim",
        "ModelDecisionOfClaim_Reason",
        "ModelBeliefClimateExists",
        "ModelBeliefClimateExists_Reason",
        "Evidence_ids",
        "Evidence_source",
        "Evidence",
        "ClaimEntropy"

    ]
    df_out = df_out[cols]
    out_path = outdir / f"{args.model}_E{args.evidenceLevel}_N{args.n_evidence}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"wrote {len(df_out)} rows to {out_path}")

if __name__ == "__main__":
    main()
