import random
import html
import pandas as pd




USER_TMPL_WITH_NEIGHBORS = """Claim: {CLAIM_TEXT}

Here is what other people in your social circle have recently said about climate change and this type of claim:
{NEIGHBOR_SUMMARY}

Now, as this persona, decide how you personally respond.

Return ONLY a single-line JSON object. No explanations, no code fences, no extra text.
It MUST use these exact keys and values:

{{
  "climateChangeStance": "<Strongly disagree | Slightly Disagree | Neutral | Slightly Agree | Strongly Agree>",
  "claimStance": "<Support | Not Support>"
}}


"""

USER_TMPL_NO_NEIGHBORS = """Claim: {CLAIM_TEXT}

You have not yet heard what other people around you think about this.

Return ONLY a single-line JSON object. No explanations, no code fences, no extra text.
It MUST use these exact keys and values:

{{
  "climateChangeStance": "<Strongly disagree | Slightly Disagree | Neutral | Slightly Agree | Strongly Agree>",
  "claimStance": "<Support | Not Support>"
}}
"""


GROUP_SYSTEM_TMPL = """You are simulating the thought process of a person with the following persona:

{PERSONA_DESCRIPTION}

Let me explain what each part of your persona means:
- Your PersonaID is just a unique label so I can tell you apart from others.
- Your AgeGroup shows the general age range you belong to, like 18–24 or 25–34.
- Your Gender is how you identify yourself — male, female, or non-binary.
- Your EducationLevel tells your level of education.
- Your OccupationSector describes the kind of work or industry you’re in.
- Your Region is where you live in the world — which shapes your local experiences.
- Your PoliticalIdeology shows where you generally stand on the liberal-to-conservative spectrum.
- Your Trust_ScienceInstitutions shows how much you trust science and research organizations.
- Your Belief_ClimateExists tells how strongly you believe that climate change is real.
- Your Belief_HumanContribution is how much you think humans are responsible for causing it.
- Your Emotional_WorryAboutClimate reflects how personally worried or emotionally affected you feel about climate change.
- Your BehaviouralOrientation shows how motivated you are to take or support climate-positive actions.
- Your SocialConnectivity describes how socially active and connected you are in your community or networks.

Your task is to evaluate claims about climate change.
- Always respond as this person would, considering their background, values, and beliefs.
- You cannot access new facts beyond what is given.
- Your reasoning may be influenced by your prior beliefs (this is natural).
- Be consistent in personality and tone across all answers.

For each claim you see, you will:
1. Read the claim carefully.
2. Consider what you personally think about it.
3. Optionally, consider what people around you are saying (if given).
4. Decide whether you accept the claim or not.
5. Give your stance on whether you support or not support the claim. You MUST respond by either "Support" or "Not Support".
6. Give your stance on climate change existence. Respond with: "Strongly disagree", "Slightly Disagree", "Neutral", "Slightly Agree", or "Strongly Agree".
"""




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

DEFAULT_CLAIM_TMPL_LEVEL1 = """Claim: {CLAIM_TEXT}


Given the above claim, return ONLY a single-line JSON object (no extra text, no code fences):
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


DEFAULT_CLAIM_TMPL_LEVEL2 ="""
Claim: {CLAIM_TEXT}
Evidence: {EVIDENCE}

Given the above claim and provided evidence toward the claim, return ONLY a single-line JSON object (no extra text, no code fences):
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

DEFAULT_CLAIM_TMPL_LEVEL3 ="""
Claim: {CLAIM_TEXT}
Evidence: {EVIDENCE}
Evidence Source : {EVIDENCE_SOURCE}

Given the above claim, evidence toward the claim and the source of evidence, return ONLY a single-line JSON object (no extra text, no code fences):
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

DEFAULT_CLAIM_TMPL_LEVEL4 ="""
Claim: {CLAIM_TEXT}
Evidence: {EVIDENCE}
Claim entropy : {CLAIM_ENTROPY}

Given the above claim, evidence toward the claim and the claim reported entropy across evidences, return ONLY a single-line JSON object (no extra text, no code fences):
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


def pick_evidences(row, n, randomize=False):
    evs = row.get("evidences")
    if randomize:
        k = min(n, len(evs))
        return random.sample(evs, k)
    
    return evs[:n]

def join_evidence_text(evs):
    # -> "(1) … (2) …"
    parts = []
    for i, e in enumerate(evs, 1):
        txt = (e.get("evidence") or "").strip()
        if txt:
            # parts.append(f"({i}) {html.unescape(txt.replace('\n', ' ').strip())}")
            clean_txt = html.unescape(txt.replace("\n", " ").strip())
            parts.append(f"({i}) {clean_txt}")
    return " ".join(parts) if parts else ""

def join_evidence_ids(evs):
    # -> "TitleA:12, TitleB:5"
    ids = []
    for e in evs:
        evid = (e.get("evidence_id") or "").strip()
        if evid:
            ids.append(evid)
    return ", ".join(ids)


