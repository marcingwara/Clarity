from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
import re


# -----------------------------
# Topic classifier (rule-based)
# -----------------------------
def _simple_topic(problem: str) -> str:
    """
    Very simple topic classifier (no ML) just to create a nice 'product' feel.
    You can replace this later with ML clustering.
    """
    text = (problem or "").lower()

    rules = [
        ("career", ["prac", "job", "work", "career", "firma", "awans", "zwoln", "rekrut", "cv", "resume", "zatrudn"]),
        ("money", ["pien", "money", "kredyt", "loan", "oszcz", "finans", "inwest", "rent", "czynsz", "budż", "salary", "pensj", "wynagrodz"]),
        ("relationships", ["zwią", "relac", "partner", "małżeń", "rodzin", "przyja", "friends", "dating"]),
        ("health", ["zdrow", "health", "trening", "gym", "dieta", "sleep", "sen", "lekarz", "terap", "psych"]),
        ("education", ["kurs", "nauk", "study", "studi", "python", "java", "ml", "ai", "cert", "szkol", "bootcamp"]),
        ("life", ["wyjaz", "wakac", "travel", "mieszkan", "przeprowadz", "dom", "car", "samoch", "auto", "kupi", "sprzeda"]),
    ]

    for label, keywords in rules:
        if any(k in text for k in keywords):
            return label
    return "other"


# -----------------------------
# Helpers: text signals
# -----------------------------
def _tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    return re.findall(r"[a-ząćęłńóśźż0-9]+", t)


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def _decision_style(decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight heuristics (no ML):
    tries to estimate the decision-maker "style" from the text.
    """
    problem = (decision.get("problem") or "")
    result_text = (decision.get("result_text") or "")
    future = (decision.get("future_reflection") or "")

    text_all = (problem + "\n" + result_text + "\n" + future).lower()

    # signals
    risk_words = ["risk", "ryzy", "obaw", "fear", "boję", "niepew", "uncertain"]
    action_words = ["next step", "one safe next step", "krok", "zrób", "wyślij", "aplik", "spróbuj", "plan"]
    overthink_words = ["nie wiem", "może", "zależy", "czasem", "co jeśli", "what if", "overthink"]

    risk_focus = 1 if _contains_any(text_all, risk_words) else 0
    action_focus = 1 if _contains_any(text_all, action_words) else 0
    overthink_signal = 1 if _contains_any(text_all, overthink_words) else 0

    # rough score buckets
    # Explorer: more action, more growth, more change language
    # Stabilizer: more risk language, stability language
    # Overthinker: uncertainty language and lack of action step
    change_words = ["zmiana", "change", "new", "nowe", "przejść", "odejść", "switch"]
    stability_words = ["stabil", "bezpie", "secure", "pewno", "spokój"]

    explorer = 1 if _contains_any(text_all, change_words) else 0
    stabilizer = 1 if _contains_any(text_all, stability_words) else 0

    # build style label
    # (keep deterministic and explainable)
    if overthink_signal and not action_focus:
        style = "Overthinker risk"
    elif explorer and action_focus and not stabilizer:
        style = "Explorer"
    elif stabilizer and risk_focus:
        style = "Stabilizer"
    else:
        style = "Balanced"

    return {
        "style": style,
        "signals": {
            "risk_focus": risk_focus,
            "action_focus": action_focus,
            "overthink_signal": overthink_signal,
        }
    }


def _habit_recommendation(agg: Dict[str, Any]) -> str:
    """
    Single, grounded habit suggestion based on aggregate signals.
    """
    styles = agg.get("style_counts", Counter())
    top_style = styles.most_common(1)[0][0] if styles else "Balanced"

    if top_style == "Overthinker risk":
        return "Before asking for more input, write ONE concrete next step and do it within 24 hours."
    if top_style == "Explorer":
        return "Add a risk-check: write 2 downside scenarios + how you'd handle them before committing."
    if top_style == "Stabilizer":
        return "Add a growth-check: define 1 small experiment (low risk) that moves you forward this week."
    return "Keep using priorities: before choosing, rank top 2 priorities and reject options that violate #1."


# -----------------------------
# Main insights
# -----------------------------
def compute_insights(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a dict with:
    - total decisions
    - priority distribution
    - topic distribution
    - top words (from problem)
    - pattern detector (priority x topic)
    - style detector (heuristics)
    - habit recommendation
    """
    total = len(decisions)

    # priorities
    pr = []
    for d in decisions:
        pr.extend(d.get("priorities") or [])
    pr_counts = Counter(pr)

    # topics
    topics = [_simple_topic(d.get("problem", "")) for d in decisions]
    topic_counts = Counter(topics)

    # top words (lightweight)
    stop = {
        "i", "a", "the", "to", "and", "or", "of", "in", "on", "for",
        "czy", "jak", "co", "w", "na", "do", "z", "że", "się", "jest", "mam",
        "praca", "pracy"
    }
    words = []
    for d in decisions:
        txt = (d.get("problem") or "").lower()
        tokens = re.findall(r"[a-ząćęłńóśźż0-9]+", txt)
        tokens = [t for t in tokens if t not in stop and len(t) >= 3]
        words.extend(tokens)
    word_counts = Counter(words)

    # -----------------------------
    # Pattern detector: priorities by topic
    # -----------------------------
    pr_by_topic = defaultdict(Counter)
    for d in decisions:
        t = _simple_topic(d.get("problem", ""))
        for p in (d.get("priorities") or []):
            pr_by_topic[t][p] += 1

    # flatten as readable "rules"
    pattern_rules = []
    for t, c in pr_by_topic.items():
        if not c:
            continue
        top = c.most_common(1)[0]
        # only show if it appears more than once or if total is tiny
        if top[1] >= 2 or total <= 3:
            pattern_rules.append({
                "topic": t,
                "top_priority": top[0],
                "count": top[1],
            })

    # -----------------------------
    # Style detector (heuristics)
    # -----------------------------
    style_counts = Counter()
    risk_total = 0
    action_total = 0
    overthink_total = 0

    for d in decisions:
        s = _decision_style(d)
        style_counts[s["style"]] += 1
        sig = s["signals"]
        risk_total += sig["risk_focus"]
        action_total += sig["action_focus"]
        overthink_total += sig["overthink_signal"]

    # normalize to simple ratios
    def ratio(x: int) -> float:
        return 0.0 if total == 0 else round(x / total, 2)

    agg = {
        "total": total,
        "priority_counts": pr_counts,
        "topic_counts": topic_counts,
        "top_words": word_counts.most_common(8),

        # v2:
        "pattern_rules": pattern_rules,
        "style_counts": style_counts,
        "style_signals": {
            "risk_focus_ratio": ratio(risk_total),
            "action_focus_ratio": ratio(action_total),
            "overthink_ratio": ratio(overthink_total),
        },
    }

    agg["habit"] = _habit_recommendation(agg)
    return agg


def format_top(counter: Counter, k: int = 3) -> List[Tuple[str, int]]:
    return counter.most_common(k)