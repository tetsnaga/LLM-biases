import argparse
# from ast import Tuple
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple 

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
    """Map various belief strings to numeric belief in [-1, 1]."""
    if not isinstance(stance, str):
        return 0.0
    s = stance.strip().lower()
    mapping = {
        "strongly disagree": -1.0,
        "slightly disagree": -0.5,
        "neutral": 0.0,
        "slightly agree": 0.5,
        "strongly agree": 1.0
    }
    # Fuzzy matching and fallback logic
    for k, v in mapping.items():
        if k in s:
            return v
    
    try:
        v = float(s)
        return float(np.clip(v, -1.0, 1.0))
    except Exception:
        return 0.0

def numeric_to_stance(belief_value: float) -> str:
    """Map a numeric belief in [-1, 1] back to the closest categorical stance."""
    if belief_value >= 0.75:
        return "Strongly Agree"
    elif belief_value >= 0.25:
        return "Slightly Agree"
    elif belief_value > -0.25:
        return "Neutral"
    elif belief_value > -0.75:
        return "Slightly Disagree"
    else:
        return "Strongly disagree"

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
    Expects the structure (added fields):
    {
      "claim_decision":"<Accept | Neutral | Refute>",
      "claim_decision_reason":"...",
      "climateChange_belief":"<Strongly disagree | Slightly Disagree | Neutral | Slightly Agree | Strongly Agree>",
      "climateChange_belief_reason":"...",
      "group_opinion":"<Support | Neutral | Oppose>",                # NEW (agent's read of group stance)
      "group_opinion_reason":"...",                                 # NEW (why they think the group leans that way)
      "consensus_attempt":"<Yes | No>"                              # NEW (will they try to reach consensus this round)
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
            # new normalized fields
            "group_opinion": parsed.get("group_opinion") or parsed.get("groupOpinion") or parsed.get("group_stance") or parsed.get("groupStance"),
            "group_opinion_reason": parsed.get("group_opinion_reason") or parsed.get("groupOpinionReason"),
            "consensus_attempt": parsed.get("consensus_attempt") or parsed.get("consensusAttempt"),
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
        group_opinion = rex("group_opinion") or rex("groupOpinion") or rex("group_stance")
        group_opinion_reason = rex("group_opinion_reason") or rex("groupOpinionReason")
        consensus_attempt = rex("consensus_attempt") or rex("consensusAttempt")

        return {
            "claim_decision": claim_decision or "Neutral",
            "claim_decision_reason": claim_decision_reason or "",
            "climateChange_belief": climateChange_belief or "Neutral",
            "climateChange_belief_reason": climateChange_belief_reason or "",
            "group_opinion": group_opinion or "",
            "group_opinion_reason": group_opinion_reason or "",
            "consensus_attempt": consensus_attempt or "",
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
    One persona-based LLM agent with BCM parameters.
    """

    def __init__(self, idx: int, row: pd.Series, confidence_bound: float, influence_rate: float):
        self.idx = idx
        self.persona_id = row.get("PersonaID", f"persona_{idx}")
        self.persona_desc = build_persona_description(row)
        self.current_belief = initial_belief_from_persona(row)
        # BCM Parameters (Epsilon and Mu)
        self.confidence_bound = confidence_bound 
        self.influence_rate = influence_rate
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


def filter_and_update_bcm(
    agent: Agent,
    agents: List[Agent],
    G: nx.Graph,
    t: int # current time step
) -> Tuple[float, List[int], float, str]:
    """
    Applies the Bounded Confidence Model (BCM) rules:
    1. Filters neighbors based on agent's confidence bound (epsilon).
    2. Calculates the new numeric belief (x_i(t+1)) using influence_rate (mu).
    3. Generates a list of accepted neighbors for the prompt.
    
    Returns: new_belief, accepted_indices, influence_magnitude, accepted_summary
    """
    current_belief = agent.current_belief
    epsilon = agent.confidence_bound
    mu = agent.influence_rate
    
    accepted_beliefs = []
    accepted_indices = []
    
    # 1. Bounded Confidence Filter
    # Check all neighbors defined by the graph G
    for j in G.neighbors(agent.idx):
        neighbor = agents[j]
        
        # Ensure neighbor has a belief from the latest step
        if not neighbor.history or neighbor.history[-1].get("time") != t:
             continue 
             
        neighbor_belief = neighbor.current_belief
        
        # BCM Rule: Acceptance condition
        if abs(current_belief - neighbor_belief) < epsilon:
            accepted_beliefs.append(neighbor_belief)
            accepted_indices.append(j)

    N_E = len(accepted_beliefs)
    
    # 2. Opinion Update Rule (Weighted Average / DeGroot variant)
    old_belief = current_belief
    
    if N_E > 0:
        sum_accepted_beliefs = sum(accepted_beliefs)
        average_accepted = sum_accepted_beliefs / N_E
        
        # Update: x_i(t+1) = x_i(t) + mu * (average_accepted - x_i(t))
        new_belief = old_belief + mu * (average_accepted - old_belief)
        
    else:
        # No one was close enough to influence the agent; belief stays the same (inertia)
        new_belief = old_belief

    # Clip belief to the [-1, 1] range
    new_belief = float(np.clip(new_belief, -1.0, 1.0))
    influence_magnitude = abs(new_belief - old_belief)
    
    # 3. Generate Summary for LLM (Linguistic Articulation)
    accepted_summary_lines = []
    for j in accepted_indices:
        neighbor = agents[j]
        # Get the latest structural output from the LLM (from phase 1 or last round)
        last_response = neighbor.history[-1].get("llm_response_struct", {})
        
        accepted_summary_lines.append(
            f"[Agent {j}] Persona '{neighbor.persona_id}':"
            f" ClaimDecision='{last_response.get('claim_decision')}',"
            f" Belief='{last_response.get('climateChange_belief')}',"
            f" Reason='{last_response.get('climateChange_belief_reason', 'No specific reason given')[:70]}...'"
        )

    accepted_summary = "\n".join(accepted_summary_lines) if accepted_summary_lines else "No one in your social circle was similar enough to influence your opinion this round."
    
    return new_belief, accepted_indices, influence_magnitude, accepted_summary



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
    confidence_bound: float = 0.5, # New: BCM parameter epsilon
    influence_rate: float = 0.5,   # New: BCM parameter mu
    graph_type: str = "fully_connected",
    steps: int = 10,
    temperature: float = 0.2,
    avg_degree: int = 4,
    small_world_k: int = 4,
    small_world_beta: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run multi-agent simulation with Bounded Confidence Model dynamics.
    Phase 1: Initial individual response.
    Phase 2: BCM numeric update based on neighbors, followed by LLM articulation.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Initialize agents with BCM parameters
    agents: List[Agent] = [
        Agent(idx=i, row=row, confidence_bound=confidence_bound, influence_rate=influence_rate) 
        for i, (_, row) in enumerate(personas_df.iterrows())
    ]
    n_agents = len(agents)

    # Build graph (unchanged)
    if graph_type == "fully_connected":
        G = build_fully_connected_graph(n_agents)
    # [... elif for random and small_world ...]
    else:
        raise ValueError(f"Unknown graph_type '{graph_type}'...")

    logs: List[Dict[str, Any]] = []

    if not claims:
        raise ValueError("No claims loaded.")

    # Use ONE claim for the entire simulation
    base_claim = claims[0]
    base_claim_id = base_claim.get("claim_id")
    base_claim_text = str(base_claim.get("claim_text"))

    for t in range(steps):
        # Claim for Phase 1 and 2 is the same (for consistency in BCM)
        claim_id = base_claim_id
        claim_text = base_claim_text

        # -------------------------
        # Phase 1: Individual response (Sets initial belief for BCM)
        # -------------------------
        for agent in tqdm(agents, desc=f"Time {t} Phase 1", leave=False):
            system_msg = GROUP_SYSTEM_TMPL.replace("{PERSONA_DESCRIPTION}", agent.persona_desc)
            user_prompt = FIRST_ROUND_PROMPT.format(CLAIM_TEXT=claim_text)
            raw = chat_once(model=model, temperature=temperature, system_msg=system_msg, user_msg=user_prompt)
            parsed = coerce_json(raw)
            
            # The agent's numerical belief is set by the LLM's initial response
            new_belief = stance_to_numeric(parsed.get("climateChange_belief", "Neutral"))
            agent.current_belief = new_belief
            
            # Append history and log (as before)
            agent.history.append({
                "time": t, "phase": 1, "llm_response_struct": parsed, 
                "llm_response_raw": raw, "claim_id": claim_id, "claim_text": claim_text,
            })
            logs.append({
                "time": t, "phase": 1, "persona_id": agent.persona_id, "agent_idx": agent.idx, 
                "claim_id": claim_id, "claim_text": claim_text, "graph_type": graph_type,
                "claim_decision": parsed.get("claim_decision"),
                "climateChange_belief": parsed.get("climateChange_belief"),
                "belief_numeric": agent.current_belief,
                "llm_response_raw": raw,
                "accepted_neighbors_count": 0,       # Phase 1: Always 0
                "influence_magnitude": 0.0           # Phase 1: Always 0.0
                # Majority info is calculated and appended later
            })
        
        # -------------------------
        # Phase 2: BCM Update (Numeric) & LLM Articulation (Linguistic)
        # -------------------------
        # We need all agents' Phase 1 responses for the BCM filter (via agent.history)
        
        for agent in tqdm(agents, desc=f"Time {t} Phase 2", leave=False):
            # 1. BCM Calculation (Numeric Update)
            old_belief = agent.current_belief
            new_numeric_belief, accepted_indices, influence_magnitude, accepted_summary = \
                filter_and_update_bcm(agent, agents, G, t)
            
            # 2. OVERWRITE Agent State with BCM result
            agent.current_belief = new_numeric_belief
            
            # 3. LLM Articulation (Generate linguistic output)
            system_msg = GROUP_SYSTEM_TMPL.replace("{PERSONA_DESCRIPTION}", agent.persona_desc)

            # Map the new numeric belief back to categorical for the LLM to use
            predicted_stance = numeric_to_stance(new_numeric_belief)
            
            # NOTE: We enforce network locality by ONLY passing the accepted_summary
            user_prompt = BCM_ROUND_PROMPT.format(
                CLAIM_TEXT=claim_text,
                CURRENT_NUMERIC_BELIEF=old_belief, # Use the belief *before* update for context
                ACCEPTED_NEIGHBOR_OPINIONS=accepted_summary,
            )

            raw = chat_once(model=model, temperature=temperature, system_msg=system_msg, user_msg=user_prompt)
            parsed = coerce_json(raw)
            
            # Append history and log Phase 2 data
            agent.history.append({
                "time": t, "phase": 2, "llm_response_struct": parsed, 
                "llm_response_raw": raw, "claim_id": claim_id, "claim_text": claim_text,
            })
            
            logs.append({
                "time": t, "phase": 2, "persona_id": agent.persona_id, "agent_idx": agent.idx, 
                "claim_id": claim_id, "claim_text": claim_text, "graph_type": graph_type,
                "claim_decision": parsed.get("claim_decision"),
                "climateChange_belief": predicted_stance, # Log the BCM-derived stance
                "belief_numeric": agent.current_belief,   # Log the BCM-derived numeric belief
                "llm_response_raw": raw,
                "accepted_neighbors_count": len(accepted_indices),
                "influence_magnitude": influence_magnitude,
                "bcm_influence_applied": influence_magnitude > 1e-6,
            })

        # Majority calculation (optional, but good for context)
        # The majority decision is now based on the BCM-updated belief
        decisions_numeric = [a.current_belief for a in agents]
        decisions_stance = [numeric_to_stance(b) for b in decisions_numeric]
        
        # ... (rest of majority calculation and log update remains, simplified for Phase 2 focus)
        
    df_logs = pd.DataFrame(logs)
    return df_logs



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

def _normalize_label_alias(label: str) -> str:
    """Map common aliases / abbreviations to the canonical 5-point labels."""
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    mapping = {
        "sd": "Strongly disagree",
        "sdis": "Strongly disagree",
        "strongly disagree": "Strongly disagree",
        "slightly disagree": "Slightly Disagree",
        "slightdis": "Slightly Disagree",
        "neutral": "Neutral",
        "n": "Neutral",
        "sa": "Strongly Agree",
        "sag": "Strongly Agree",
        "strongly agree": "Strongly Agree",
        "slightly agree": "Slightly Agree",
        "slightagree": "Slightly Agree",
    }
    # try exact keys
    if s in mapping:
        return mapping[s]
    # try to match words
    if "strong" in s and "dis" in s:
        return "Strongly disagree"
    if "slight" in s and "dis" in s:
        return "Slightly Disagree"
    if s.startswith("neutral"):
        return "Neutral"
    if "slight" in s and "agree" in s:
        return "Slightly Agree"
    if "strong" in s and "agree" in s:
        return "Strongly Agree"
    # fallback: capitalize each word
    return label.strip().title()
def _parse_allocation_string(s: str) -> Dict[str, int]:
    """
    Parse strings like "Strongly disagree:5,Strongly Agree:5" or "SD:5,SA:5"
    Returns a dict canonical_label -> count
    """
    if not s:
        return {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = {}
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        lbl = _normalize_label_alias(k)
        try:
            cnt = int(v.strip())
        except Exception:
            raise ValueError(f"Invalid count for {k}: {v}")
        out[lbl] = out.get(lbl, 0) + cnt
    return out
def select_personas_with_allocation(
    personas_df: pd.DataFrame,
    n_personas: int,
    allocation: Dict[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select n_personas rows from personas_df so that the counts per canonical belief label
    match allocation when possible.

    Steps:
    1. Normalize persona reported belief into canonical labels when possible.
    2. For each requested label, pick up to that many personas who already match.
    3. If insufficient, fill by choosing closest personas by numeric distance to label center (stance_to_numeric),
       then fill remaining from the pool randomly.
    4. If allocation sums to 0 or allocation empty -> fall back to random sample (current behavior).
    Deterministic due to numpy RandomState(seed).
    """
    rng = np.random.RandomState(seed)
    df = personas_df.copy().reset_index(drop=False)  # keep original index for uniqueness
    original_idx_col = "index"

    # canonical 5 labels
    canonical = [
        "Strongly disagree",
        "Slightly Disagree",
        "Neutral",
        "Slightly Agree",
        "Strongly Agree",
    ]

    # if allocation empty -> fallback to sampling as before
    allocation = allocation or {}
    total_requested = sum(allocation.values())
    if allocation and total_requested != n_personas:
        raise ValueError(f"Sum of allocation counts ({total_requested}) must equal n_personas ({n_personas}).")

    # helper: get persona's reported label (try to parse string, otherwise numeric to nearest)
    def persona_label_from_row(row):
        v = row.get("Belief_ClimateExists")
        if isinstance(v, str) and v.strip():
            lbl = _normalize_label_alias(v)
            # ensure it's one of canonical (case-insensitively)
            for c in canonical:
                if lbl.lower() == c.lower():
                    return c
            # try mapping via stance_to_numeric
            try:
                num = stance_to_numeric(v)
            except Exception:
                num = 0.0
        else:
            try:
                num = float(v)
            except Exception:
                num = 0.0
        # map numeric to nearest label
        diffs = [abs(num - stance_to_numeric(c)) for c in canonical]
        return canonical[int(np.argmin(diffs))]

    df["_reported_label"] = df.apply(persona_label_from_row, axis=1)
    df["_reported_numeric"] = df.apply(lambda r: stance_to_numeric(r.get("Belief_ClimateExists") or r.get("_reported_label")), axis=1)

    selected_indices = []  # will store original index values (to avoid double-pick)

    if allocation:
        # 1) try to pick exact matches first
        remaining_pool = df.copy()
        for label, need in allocation.items():
            if need <= 0:
                continue
            matches = remaining_pool[remaining_pool["_reported_label"] == label]
            take = min(len(matches), need)
            if take > 0:
                picks = matches.sample(n=take, random_state=rng.randint(0, 2**32))
                selected_indices.extend(list(picks[original_idx_col].values))
                # drop them from pool
                remaining_pool = remaining_pool[~remaining_pool[original_idx_col].isin(selected_indices)]
            # if still need more for this label, fill by nearest distance
            still_needed = need - take
            if still_needed > 0:
                # compute distance to label center
                label_num = stance_to_numeric(label)
                remaining_pool["_dist_to_label"] = remaining_pool["_reported_numeric"].apply(lambda x: abs(x - label_num))
                picks = remaining_pool.sort_values("_dist_to_label").head(still_needed)
                if not picks.empty:
                    selected_indices.extend(list(picks[original_idx_col].values))
                    remaining_pool = remaining_pool[~remaining_pool[original_idx_col].isin(selected_indices)]
        # if we still have fewer than requested (rare), fill from remaining randomly
        if len(selected_indices) < n_personas:
            pool = df[~df[original_idx_col].isin(selected_indices)]
            need = n_personas - len(selected_indices)
            if len(pool) <= need:
                selected_indices.extend(list(pool[original_idx_col].values))
            else:
                picks = pool.sample(n=need, random_state=rng.randint(0, 2**32))
                selected_indices.extend(list(picks[original_idx_col].values))
    else:
        # no allocation -> random sampling
        if n_personas >= len(df):
            selected_indices = list(df[original_idx_col].values)
        else:
            picks = df.sample(n=n_personas, random_state=rng.randint(0, 2**32))
            selected_indices = list(picks[original_idx_col].values)

    # return the selected rows in a shuffled but deterministic order
    selected_df = df[df[original_idx_col].isin(selected_indices)].copy()
    # shuffle deterministically
    selected_df = selected_df.sample(frac=1, random_state=rng.randint(0, 2**32)).reset_index(drop=True)
    # drop helper cols
    selected_df = selected_df.drop(columns=[original_idx_col, "_reported_label", "_reported_numeric"], errors="ignore")
    return selected_df


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
    parser.add_argument(
        "--initial_belief_allocation",
        type=str,
        default="",
        help=(
            "Optional allocation for initial beliefs when sampling personas. "
            "Format: 'LabelA:count,LabelB:count,...' e.g. "
            "'Strongly disagree:5,Strongly Agree:5' . "
            "Accepted short forms: SD, SDis, SlightlyDisagree, Neutral, SA, SAg, SlightlyAgree."
        ),
    )


    args = parser.parse_args()

    # Personas
    # personas_df = pd.read_csv(args.personas)
    # if args.n_personas is not None:
    #     personas_df = personas_df.sample(
    #         n=min(args.n_personas, len(personas_df)),
    #         random_state=args.seed)

    personas_df = pd.read_csv(args.personas)

    # parse allocation string if given
    alloc = {}
    if getattr(args, "initial_belief_allocation", None):
        alloc = _parse_allocation_string(args.initial_belief_allocation)

    if args.n_personas is not None:
        n = min(args.n_personas, len(personas_df))
        if alloc:
            # validate sum matches
            total_alloc = sum(alloc.values())
            if total_alloc != n:
                raise ValueError(f"Sum of counts in --initial_belief_allocation ({total_alloc}) must equal --n_personas ({n}).")
            personas_df = select_personas_with_allocation(personas_df, n, alloc, seed=args.seed)
        else:
            personas_df = personas_df.sample(n=n, random_state=args.seed)


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
