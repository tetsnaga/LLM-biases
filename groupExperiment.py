import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans

import ollama
from prompts import *

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


def stance_to_numeric(stance: str) -> float:
    """
    Map various belief strings to numeric belief in [-1, 1].
    Accepts either Likert climateChangeStance strings or the new
    climateChange_belief labels.
    """
    if not isinstance(stance, str):
        return 0.0
    s = stance.strip().lower()
    mapping = {
        "strongly disagree.": -1.0,
        "slightly disagree.": -0.5,
        "neutral.": 0.0,
        "slightly agree.": 0.5,
        "strongly agree.": 1.0
    }

    if s in mapping:
        return mapping[s]
    # try some fuzzy matching
    if "strong" in s and "disagree" in s:
        return -1.0
    if "slight" in s and "disagree" in s:
        return -0.5
    if s.startswith("neutral"):
        return 0.0
    if "slight" in s and "agree" in s:
        return 0.5
    if "strong" in s and "agree" in s:
        return 1.0
    # fallback numeric parse
    try:
        v = float(s)
        return float(np.clip(v, -1.0, 1.0))
    except Exception:
        return 0.0


def initial_belief_from_persona(row: pd.Series) -> float:
    val = row.get("Belief_ClimateExists")
    if isinstance(val, str):
        return stance_to_numeric(val)
    v = float(val)
    return float(np.clip(v, -1.0, 1.0))



def load_claims_with_label_JSON(path: str) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    out = []
    for it in raw:
        claim_text = it.get("claim")
        if not claim_text:
            raise ValueError(f"Claim text not found in item: {it}")
        
        out.append({
            "claim_id": it.get("claim_id") or it.get("id"),
            "claim_text": claim_text,
            "claim_stance_label": it.get("label")
            or it.get("claim_label")
            or it.get("verdict")
            or it.get("stance"),
        })
    return out


def load_claims_with_label_CSV(path: str) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(Path(path))
    except Exception as e:
        print(f"Error reading CSV file at {path}: {e}")
        return []

    raw = df.to_dict("records")
    out = []
    for it in raw:
        claim_text = (
            it.get("claim") or it.get("claim_text") or it.get("statement")
            or it.get("text") or ""
        )
        if not claim_text:
            continue

        out.append({
            "claim_id": it.get("claim_id") or it.get("id"),
            "claim_text": claim_text,
            "claim_stance_label": it.get("stance_label")
        })
    return out


def filter_balanced_claims(claims: List[Dict[str, Any]], n_each: int = 100) -> List[Dict[str, Any]]:
    df = pd.DataFrame(claims)
    if "claim_stance_label" not in df.columns:
        raise ValueError("Expected 'claim_stance_label' field in claims.")

    df["claim_stance_label"] = df["claim_stance_label"].astype(str).str.upper()

    supports = df[df["claim_stance_label"] == "SUPPORTS"]
    refutes = df[df["claim_stance_label"] == "REFUTES"]

    n_each = min(n_each, len(supports), len(refutes))
    supports_sample = supports.sample(n=n_each, random_state=42)
    refutes_sample = refutes.sample(n=n_each, random_state=42)

    balanced = pd.concat([supports_sample, refutes_sample]).sample(
        frac=1, random_state=42
    ).to_dict(orient="records")
    return balanced


def coerce_json(text: str) -> Dict[str, Any]:
    """
    Forgiving JSON parser for the agent's response.
    Expects the structure:
    {
      "claim_decision":"<Accept | Neutral | Refute>",
      "claim_decision_reason":"...",
      "climateChange_belief":"<Strongly disagree | Slightly Disagree | Neutral | Slightly Agree | Strongly Agree>",
      "climateChange_belief_reason":"..."
    }
    Falls back to extracting fields by regex if strict JSON parsing fails.
    """
    text_clean = text.strip().strip("`").replace("```json", "").replace("```", "")
    s, e = text_clean.find("{"), text_clean.rfind("}")
    if s != -1 and e != -1:
        text_clean = text_clean[s:e + 1]

    try:
        parsed = json.loads(text_clean)
        # normalize keys (some prompts may return slight variants)
        out = {
            "claim_decision": parsed.get("claim_decision") or parsed.get("claimDecision") or parsed.get("claim_stance") or parsed.get("claimStance"),
            "claim_decision_reason": parsed.get("claim_decision_reason") or parsed.get("claimDecisionReason") or parsed.get("claim_decision_reasoning") or parsed.get("reason"),
            "climateChange_belief": parsed.get("climateChange_belief") or parsed.get("climateChange_belief_label") or parsed.get("climateChangeBelief") or parsed.get("climateChangeStance"),
            "climateChange_belief_reason": parsed.get("climateChange_belief_reason") or parsed.get("climateChangeBeliefReason") or parsed.get("climateChange_belief_reasoning"),
        }
        # fill defaults
        out = {k: (v if v is not None else "") for k, v in out.items()}
        return out
    except Exception:
        # fallback: try to extract fields via regex
        def rex(key):
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text_clean, re.IGNORECASE)
            return m.group(1) if m else None

        claim_decision = rex("claim_decision") or rex("claimDecision") or rex("claim_stance") or rex("claimStance")
        claim_decision_reason = rex("claim_decision_reason") or rex("claimDecisionReason") or rex("reason")
        climateChange_belief = rex("climateChange_belief") or rex("climateChangeStance") or rex("climateChangeBelief")
        climateChange_belief_reason = rex("climateChange_belief_reason") or rex("climateChangeBeliefReason")

        return {
            "claim_decision": claim_decision or "Neutral",
            "claim_decision_reason": claim_decision_reason or "",
            "climateChange_belief": climateChange_belief or "Neutral",
            "climateChange_belief_reason": climateChange_belief_reason or "",
        }


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


def chat_seq(model: str, temperature: float, messages: List[Dict[str, str]]) -> str:
    r = ollama.chat(
        model=model,
        options={"temperature": temperature},
        messages=messages,
    )
    return r["message"]["content"].strip()



class Agent:
    """
    One persona-based LLM agent.
    Belief is inferred from last climateChange_belief.
    """

    def __init__(self, idx: int, row: pd.Series):
        self.idx = idx
        self.persona_id = row.get("PersonaID", f"persona_{idx}")
        self.persona_desc = build_persona_description(row)
        self.current_belief = initial_belief_from_persona(row)
        self.history: List[Dict[str, Any]] = []  # one entry per phase (phase1/phase2) per time step

    def last_response(self) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None
        return self.history[-1].get("llm_response_struct")


def build_fully_connected_graph(n_agents: int) -> nx.Graph:
    """
    Every agent sees every other agent.
    Neutral, maximally mixed environment.
    """
    G = nx.complete_graph(n_agents)
    return G


def build_random_graph(n_agents: int, avg_degree: int = 4, seed: int = 42) -> nx.Graph:
    """
    Erdős–Rényi random graph, no bias from beliefs.
    """
    p = avg_degree / max(n_agents - 1, 1)
    G = nx.erdos_renyi_graph(n_agents, p, seed=seed)

    # ensure connectivity (optional but nice)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for c1, c2 in zip(components[:-1], components[1:]):
            i = next(iter(c1))
            j = next(iter(c2))
            G.add_edge(i, j)
    return G


def build_small_world_graph(
    n_agents: int, k: int = 4, beta: float = 0.1, seed: int = 42
) -> nx.Graph:
    """
    Watts–Strogatz small-world graph.
    Also neutral w.r.t. beliefs.
    k = each node is connected to k nearest neighbors in a ring (must be even).
    beta = rewiring probability.
    """
    if k % 2 == 1:
        k += 1  # enforce even
    if k >= n_agents:
        k = max(2, n_agents - 1)
        if k % 2 == 1:
            k -= 1
    G = nx.watts_strogatz_graph(n_agents, k, beta, seed=seed)
    return G


def summarize_neighbors_all(
    agent_idx: int,
    agents: List[Agent],
    G: nx.Graph,
) -> str:
    """
    Return a multi-line string containing ALL neighbors' Phase-1 (or latest) responses:
    Each line includes neighbor id, name, claim_decision and its reason, and climateChange_belief and its reason.

    If no neighbor has expressed an opinion yet, returns an explicit sentence.
    """
    neighbors = list(G.neighbors(agent_idx))
    if not neighbors:
        return "No one in your social circle has expressed an opinion yet."

    # Keep ordering deterministic for reproducibility
    neighbors = sorted(neighbors)

    lines = []
    any_with_history = False
    for j in neighbors:
        if not agents[j].history:
            # skip neighbors who never responded yet
            continue
        any_with_history = True
        # take the last recorded response struct (could be phase1 or phase2)
        h = agents[j].history[-1]
        llm_struct = h.get("llm_response_struct") or {}
        cdecision = llm_struct.get("claim_decision", "Neutral")
        cdecision_reason = llm_struct.get("claim_decision_reason", "").strip()
        cbelief = llm_struct.get("climateChange_belief", "Neutral")
        cbelief_reason = llm_struct.get("climateChange_belief_reason", "").strip()

        lines.append(
            f"[Neighbor {j}] {agents[j].persona_id}: claim_decision='{cdecision}' (why: {cdecision_reason}) ; "
            f"belief='{cbelief}' (why: {cbelief_reason})"
        )

    if not any_with_history:
        return "No one in your social circle has expressed an opinion yet."

    return "\n".join(lines)

def simulate_polarization(
    model: str,
    personas_df: pd.DataFrame,
    claims: List[Dict[str, Any]],
    graph_type: str = "fully_connected",
    steps: int = 10,
    temperature: float = 0.2,
    avg_degree: int = 4,
    small_world_k: int = 4,
    small_world_beta: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run multi-agent simulation with two-phase rounds:
    Phase 1: give claim[t] to all agents, collect responses.
    Phase 2: give each agent the full set of Phase 1 responses (including their own),
             then present claim[t+1] (or same if not available) and collect follow-up responses.
    Returns a log DataFrame (one row per agent per phase per time step).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Initialize agents
    agents: List[Agent] = [
        Agent(idx=i, row=row) for i, (_, row) in enumerate(personas_df.iterrows())
    ]
    n_agents = len(agents)

    # Build graph
    if graph_type == "fully_connected":
        G = build_fully_connected_graph(n_agents)
    elif graph_type == "random":
        G = build_random_graph(n_agents, avg_degree=avg_degree, seed=seed)
    elif graph_type == "small_world":
        G = build_small_world_graph(
            n_agents, k=small_world_k, beta=small_world_beta, seed=seed)
    else:
        raise ValueError(f"Unknown graph_type '{graph_type}'. Use one of: fully_connected, random, small_world")

    logs: List[Dict[str, Any]] = []

    if not claims:
        raise ValueError("No claims loaded.")

    # We'll iterate steps times; each step uses claim t for Phase1 and claim t+1 for Phase2 (wrap-around)
    for t in range(steps):
        claim = claims[t % len(claims)]
        claim_id = claim.get("claim_id")
        claim_text = str(claim.get("claim_text"))

        # -------------------------
        # Phase 1: give claim to all agents and collect responses
        # -------------------------
        phase1_responses: Dict[int, Dict[str, Any]] = {}
        for agent in tqdm(agents, desc=f"Time {t} Phase 1", leave=False):
            system_msg = GROUP_SYSTEM_TMPL.replace("{PERSONA_DESCRIPTION}", agent.persona_desc)

            user_prompt = FIRST_ROUND_PROMPT.format(
                CLAIM_TEXT=claim_text)

            print("phase 1 prompt for time ", t, " agent ", agent.idx, user_prompt)
            raw = chat_once(
                model=model,
                temperature=temperature,
                system_msg=system_msg,
                user_msg=user_prompt
            )

            parsed = coerce_json(raw)
            # numeric belief
            new_belief = stance_to_numeric(parsed.get("climateChange_belief", "Neutral"))
            agent.current_belief = new_belief

            # record response struct
            phase1_responses[agent.idx] = {
                "agent_idx": agent.idx,
                "persona_id": agent.persona_id,
                "claim_id": claim_id,
                "claim_text": claim_text,
                "llm_response_raw": raw,
                "llm_response_struct": parsed,
                "belief_numeric": agent.current_belief,
            }

            # append to agent history
            agent.history.append({
                "time": t,
                "phase": 1,
                "llm_response": parsed,
                "llm_response_struct": parsed,
                "llm_response_raw": raw,
                "claim_id": claim_id,
                "claim_text": claim_text,
            })

            # log row for phase 1 (we'll augment with majority info after collecting all)
            logs.append({
                "time": t,
                "phase": 1,
                "persona_id": agent.persona_id,
                "agent_idx": agent.idx,
                "claim_id": claim_id,
                "claim_text": claim_text,
                "graph_type": graph_type,
                "claim_decision": parsed.get("claim_decision"),
                "claim_decision_reason": parsed.get("claim_decision_reason"),
                "climateChange_belief": parsed.get("climateChange_belief"),
                "climateChange_belief_reason": parsed.get("climateChange_belief_reason"),
                "belief_numeric": agent.current_belief,
                "llm_response_raw": raw,
            })

        # After Phase 1 for this time t, compute majority (>2/3) of claim_decision
        decisions = [phase1_responses[i]["llm_response_struct"].get("claim_decision", "Neutral") for i in range(n_agents)]
        # normalize decisions (capitalization)
        decisions_clean = [d.strip().title() if isinstance(d, str) else "Neutral" for d in decisions]
        counts = {}
        for d in decisions_clean:
            counts[d] = counts.get(d, 0) + 1
        # find top decision
        top_decision, top_count = None, 0
        for k, v in counts.items():
            if v > top_count:
                top_decision, top_count = k, v
        majority_reached = (top_count > (2/3) * n_agents)
        # Update the last n_agents logs entries (phase1 rows) with majority info
        for i in range(len(logs) - n_agents, len(logs)):
            logs[i]["majority_decision"] = top_decision
            logs[i]["majority_count"] = top_count
            logs[i]["majority_reached"] = bool(majority_reached)

        # -------------------------
        # Phase 2: give everyone the Phase1 responses (including themselves), and present next claim
        # -------------------------
        next_claim = claims[(t + 1) % len(claims)]
        next_claim_id = next_claim.get("claim_id")
        next_claim_text = str(next_claim.get("claim_text"))

        # Build a textual summary of phase1 responses to pass to agents
        phase1_summary_lines = []
        for idx in range(n_agents):
            r = phase1_responses[idx]["llm_response_struct"]
            persona_id = phase1_responses[idx]["persona_id"]
            cd = r.get("claim_decision", "Neutral")
            cd_reason = r.get("claim_decision_reason", "")
            cb = r.get("climateChange_belief", "Neutral")
            cb_reason = r.get("climateChange_belief_reason", "")
            phase1_summary_lines.append(
                f"{persona_id}: claim_decision='{cd}' reason='{cd_reason}' | belief='{cb}' reason='{cb_reason}'"
            )
        phase1_summary_text = "\n".join(phase1_summary_lines)

        for agent in tqdm(agents, desc=f"Time {t} Phase 2", leave=False):
            system_msg = GROUP_SYSTEM_TMPL.replace("{PERSONA_DESCRIPTION}", agent.persona_desc)

            neighbor_opinions = summarize_neighbors_all(agent.idx, agents, G)
            
            user_prompt = EACH_ROUND_PROMPT.format(
                CLAIM_TEXT=next_claim_text,
                NEIGHBOUR_ID="",
                NEIGHBOR_OPINIONS=neighbor_opinions
            )

            print("here is the prompt in time ", t, " for agent ", agent.idx, user_prompt)

            raw = chat_once(
                model=model,
                temperature=temperature,
                system_msg=system_msg,
                user_msg=user_prompt
            )

            parsed = coerce_json(raw)
            new_belief = stance_to_numeric(parsed.get("climateChange_belief", "Neutral"))
            agent.current_belief = new_belief

            # append to history
            agent.history.append({
                "time": t,
                "phase": 2,
                "llm_response": parsed,
                "llm_response_struct": parsed,
                "llm_response_raw": raw,
                "claim_id": next_claim_id,
                "claim_text": next_claim_text,
                "received_phase1_summary": phase1_summary_text,
            })

            # log phase2 row
            logs.append({
                "time": t,
                "phase": 2,
                "persona_id": agent.persona_id,
                "agent_idx": agent.idx,
                "claim_id": next_claim_id,
                "claim_text": next_claim_text,
                "graph_type": graph_type,
                "claim_decision": parsed.get("claim_decision"),
                "claim_decision_reason": parsed.get("claim_decision_reason"),
                "climateChange_belief": parsed.get("climateChange_belief"),
                "climateChange_belief_reason": parsed.get("climateChange_belief_reason"),
                "belief_numeric": agent.current_belief,
                "llm_response_raw": raw,
                # include the Phase 1 majority for convenience
                "phase1_majority_decision": top_decision,
                "phase1_majority_count": top_count,
                "phase1_majority_reached": bool(majority_reached),
            })

    df_logs = pd.DataFrame(logs)
    return df_logs


# ============================================================
# 8. Polarization metrics
# ============================================================

def compute_polarization_metrics(df_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple polarization metrics per time step from df_logs (we aggregate by time, using phase 2 belief values if available).
    """
    metrics = []
    # we compute metrics per time step using the phase 2 belief where possible, otherwise phase 1
    for t, df_t in df_logs.groupby("time"):
        # prefer phase 2 rows to get updated beliefs
        df_pref = df_t[df_t["phase"] == 2]
        if df_pref.empty:
            df_pref = df_t

        beliefs = df_pref["belief_numeric"].fillna(0).values.reshape(-1, 1)
        mean = float(beliefs.mean())
        std = float(beliefs.std())
        frac_extreme = float(np.mean(np.abs(beliefs) > 0.8))

        # cluster gap (two-cluster separation)
        if len(beliefs) >= 4:
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            km.fit(beliefs)
            m1, m2 = km.cluster_centers_.flatten()
            cluster_gap = float(abs(m1 - m2))
        else:
            cluster_gap = float("nan")

        metrics.append({
            "time": t,
            "mean_belief": mean,
            "std_belief": std,
            "frac_extreme": frac_extreme,
            "cluster_gap": cluster_gap,
        })

    return pd.DataFrame(metrics)


# ============================================================
# 9. CLI main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-agent climate polarization simulation with LLM personas.")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--personas", required=True, help="Path to personas CSV file")
    parser.add_argument("--claims", required=True, help="Path to claims CSV or JSON file")
    parser.add_argument(
        "--claims_format",
        choices=["csv", "json"],
        default="csv",
        help="Format of claims file",
    )
    parser.add_argument(
        "--graph_type",
        choices=["fully_connected", "random", "small_world"],
        default="fully_connected",  # explicit default fully connected
        help="Interaction graph type",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of time steps")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--avg_degree", type=int, default=4, help="Avg degree for random graph")
    parser.add_argument("--small_world_k", type=int, default=4, help="k for small-world")
    parser.add_argument("--small_world_beta", type=float, default=0.1, help="beta for small-world")
    parser.add_argument("--n_personas", type=int, default=None, help="Subset of personas")
    parser.add_argument("--n_claims", type=int, default=50, help="Number of claims to use")
    parser.add_argument("--balanced_claims", action="store_true",
                        help="If set, balance SUPPORTS / REFUTES (only for labeled datasets)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_prefix", default="group_polarization", help="Output prefix")

    args = parser.parse_args()

    # Personas
    personas_df = pd.read_csv(args.personas)
    if args.n_personas is not None:
        personas_df = personas_df.sample(
            n=min(args.n_personas, len(personas_df)),
            random_state=args.seed)

    # Claims
    if args.claims_format == "csv":
        claims = load_claims_with_label_CSV(args.claims)
    else:
        claims = load_claims_with_label_JSON(args.claims)

    if args.balanced_claims:
        claims = filter_balanced_claims(claims, n_each=args.n_claims // 2)
    else:
        # Just sample without balancing
        if args.n_claims is not None and args.n_claims < len(claims):
            claims = (
                pd.DataFrame(claims)
                .sample(n=args.n_claims, random_state=args.seed)
                .to_dict(orient="records"))

    print(f"Loaded {len(personas_df)} personas and {len(claims)} claims.")

    df_logs = simulate_polarization(
        model=args.model,
        personas_df=personas_df,
        claims=claims,
        graph_type=args.graph_type,
        steps=args.steps,
        temperature=args.temperature,
        avg_degree=args.avg_degree,
        small_world_k=args.small_world_k,
        small_world_beta=args.small_world_beta,
        seed=args.seed,
    )

    df_metrics = compute_polarization_metrics(df_logs)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_path = out_dir / f"{args.out_prefix}_logs.jsonl"
    metrics_path = out_dir / f"{args.out_prefix}_metrics.csv"

    # Save logs as JSONL
    with logs_path.open("w", encoding="utf-8") as f:
        for _, row in df_logs.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    # Save metrics
    df_metrics.to_csv(metrics_path, index=False)

    print(f"Saved logs to {logs_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
