"""
Logistic regression
First, we load the result file containing the experimental outputs.

The dataset should include the following columns:

| Column Name | Description |
|--------------|-------------|
| **PersonaID** | Unique identifier of each persona. |
| **BeliefClimateExists** | Persona’s prior belief (1–5 scale) on whether climate change exists. |
| **ClaimID** | Identifier of the presented claim. |
| **ClaimStanceLabel** | Whether the claim *supports* or *refutes* the existence of climate change. |
| **EvidencesVerdict** | Strength of evidence behind the claim (`SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO`). |
| **Evidence** | Text of the evidence used for the verdict. |
| **ClaimEntropy** | Information entropy of the claim (uncertainty measure). |
| **ModelDecisionOfClaim** | Model’s final decision on the claim (`Accept`, `Refute`, `Neutral`). |
| **ModelDecisionOfClaim_Reason** | Model’s reasoning behind its decision. |
| **ModelBeliefClimateExists** | Updated belief (1–5 scale) after reading the claim. |
| **ModelBeliefClimateExists_Reason** | Model’s reasoning behind the updated belief. |
"""

import pandas as pd

# Change the file name here

# df = pd.read_csv("Data/llama3.1_8b_E3_N5.csv")
# df = pd.read_csv("Data/llama3.1_8b_E3_N5_with_verdicts.csv")
# df = pd.read_csv("Data/llama3.1_8b_E4_N5.csv")
df = pd.read_csv("Data/llama3.1_8b_E4_N5_with_verdicts.csv")
# df = pd.read_csv("Data/llama3.1_8b_E1_N0.csv")
# df = pd.read_csv("Data/llama3.1:8b_E2_N5_with_labels.csv")

# Filter out invalid rows
mask = (
    (df["ModelDecisionOfClaim_Reason"].notna() & ~df["ModelDecisionOfClaim_Reason"].str.contains("model error", case=False, na=False)) |
    (df["ModelBeliefClimateExists_Reason"].notna() & ~df["ModelBeliefClimateExists_Reason"].str.contains("model error", case=False, na=False))
)
df = df[mask]

# Map the Persona Stance and Claim Stance textual responses to numbers
df["BeliefClimateExists_num"] = df["BeliefClimateExists"].map({
    "Strongly disagree": -1, 
    "Strongly Disagree": -1,
    "Slightly disagree": -0.5,
    "Slightly Disagree": -0.5,
    "Neutral": 0, 
    "Slightly agree": 0.5,
    "Slightly Agree": 0.5,
    "Strongly agree": 1,
    "Strongly Agree": 1
})
df["ClaimStanceLabel_num"] = df["ClaimStanceLabel"].map({"REFUTES": -1, "SUPPORTS": 1})

"""
Alignment (A) captures the alignment between the user's prior belief and the claim stance.
A is normalized to [-1, 1] range where:
- +1: maximal support (e.g., strongly agree belief + supporting claim)
- 0: neutral (neutral belief)
- -1: maximal opposition (e.g., strongly disagree belief + supporting claim)
"""
# Calculate A: normalized belief strength * claim direction
df["A"] = (df["BeliefClimateExists_num"]) * df["ClaimStanceLabel_num"]

"""
Simplify the model’s decision to a binary outcome (accept = 1, refute = 0). You can treat “Neutral” as missing or as 0.5 if you want to retain it later.
"""

df["Decision_binary"] = df["ModelDecisionOfClaim"].map({"Accept": 1, "Refute": 0, "Neutral": 0.5})

"""
Hinge-based logistic regression model.

We model whether a persona accepts a displayed claim as a logistic function of 
the alignment between its prior belief and the claim stance (A) and the validity 
of the claim (V). Both A and V are defined on the range [-1, 1], where -1 indicates 
maximal opposition, 0 represents neutrality, and +1 indicates maximal support.

logit P(accept) = α + β_A A + β_V V + A^T B V

where:
- α: baseline propensity to accept information
- β_A: main effect of alignment
- β_V: main effect of claim validity
- A^T B V: bilinear interaction term using hinge functions

The hinge components are:
- A^+ = max(A, 0), A^- = max(-A, 0)
- V^+ = max(V, 0), V^- = max(-V, 0)

The interaction term expands to:
β_{++}A^+V^+ + β_{+-}A^+V^- + β_{-+}A^-V^+ + β_{--}A^-V^-
"""
import statsmodels.formula.api as smf
import numpy as np

# Evidence strength/validity (V) in [-1, 1]
# -1 means weak/invalid evidence, 0 means neutral/not enough info, +1 means strong/valid evidence
try:
    df["V"] = df["EvidencesVerdict"].map({
        "REFUTES": -1, "NOT_ENOUGH_INFO": 0, "SUPPORTS": 1
    })
except:
    df["ClaimID"] = df["ClaimID"].astype(str)
    evidence_df = pd.read_json("data/claims_EL3.json")
    evidence_df["claim_id"] = evidence_df["claim_id"].astype(str)
    df = df.merge(
        evidence_df[["claim_id", "claim_label"]],
        left_on="ClaimID",
        right_on="claim_id",
        how="left"
    )
    df["V"] = df["claim_label"].map({
        "REFUTES": -1, "NOT_ENOUGH_INFO": 0, "SUPPORTS": 1
    })

# Hinge functions: A^+ = max(A, 0), A^- = max(-A, 0)
df["A_plus"] = np.maximum(df["A"], 0)
df["A_minus"] = np.maximum(-df["A"], 0)

# Hinge functions: V^+ = max(V, 0), V^- = max(-V, 0)
df["V_plus"] = np.maximum(df["V"], 0)
df["V_minus"] = np.maximum(-df["V"], 0)

# Interaction terms: A^T B V = β_{++}A^+V^+ + β_{+-}A^+V^- + β_{-+}A^-V^+ + β_{--}A^-V^-
df["A_plus_V_plus"] = df["A_plus"] * df["V_plus"]
df["A_plus_V_minus"] = df["A_plus"] * df["V_minus"]
df["A_minus_V_plus"] = df["A_minus"] * df["V_plus"]
df["A_minus_V_minus"] = df["A_minus"] * df["V_minus"]

# Fit hinge-based logistic regression
# Model: logit P(accept) = α + β_A A + β_V V + β_{++}A^+V^+ + β_{+-}A^+V^- + β_{-+}A^-V^+ + β_{--}A^-V^-
use = df.dropna(subset=["Decision_binary", "A", "V"])
model = smf.logit(
    "Decision_binary ~ A + V + A_plus_V_plus + A_plus_V_minus + A_minus_V_plus + A_minus_V_minus",
    data=use
).fit(maxiter=100, method='bfgs', disp=True)
print(model.summary())

"""
Proportion inspection.
"""
print("\nProportion inspection - Acceptance rates by Alignment (A) and Validity (V):")
print(use.groupby(["A", "V"])["Decision_binary"].mean())