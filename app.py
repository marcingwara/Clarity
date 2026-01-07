import base64
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai

from prompts import questions_prompt, analysis_prompt, future_you_prompt
from memory import load_decisions, append_decision, delete_decision_by_timestamp, rewrite_all_decisions
from ml import embed_text, top_k_similar
from insights import compute_insights, format_top

# Optional ML Topics (if topics_ml.py exists)
TOPICS_ML_AVAILABLE = False
try:
    from topics_ml import build_topic_centroids, classify_from_embedding
    TOPICS_ML_AVAILABLE = True
except Exception:
    TOPICS_ML_AVAILABLE = False

# ===================== CONFIG =====================
load_dotenv()
APP_TITLE = "🧠 Clarity"
CHAT_MODEL = "models/gemini-flash-lite-latest"
COOLDOWN_SECONDS = 1.0

# Hidden admin switches (set via env or Streamlit Secrets)
ADMIN_MODE = (os.getenv("CLARITY_ADMIN", "0") == "1")
ENABLE_ML_TOPICS_UI = (os.getenv("CLARITY_ML_TOPICS", "0") == "1")

# ===================== STREAMLIT SETUP =====================

def render_logo(path: str, width: int = 260, lift_px: int = 28):
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            justify-content:flex-start;
            margin-top: -{lift_px}px;
            margin-bottom: 8px;
        ">
            <img src="data:image/png;base64,{b64}"
                 style="width:{width}px; height:auto; display:block;" />
        </div>
        """,
        unsafe_allow_html=True
    )

render_logo("logo.png", width=480)





# ===================== API KEY =====================
def get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("Missing GEMINI_API_KEY. Add it to Streamlit Secrets (or .env locally).")
    st.stop()

client = genai.Client(api_key=api_key)

# ===================== SESSION STATE =====================
def init_state():
    defaults = {
        "stage": "questions",          # questions -> analysis
        "problem": "",
        "questions": [],
        "answers": [],
        "priorities": ["Growth"],
        "similar": [],
        "last_api_call": 0.0,

        # results
        "last_result_text": None,
        "last_saved_timestamp": None,

        # history browsing
        "history_selected_ts": None,

        # topics ml
        "topic_centroids": None,
        "use_ml_topics": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ===================== COOLDOWN (SAFE) =====================
def cooldown_ready(seconds: float = COOLDOWN_SECONDS) -> bool:
    """
    IMPORTANT: never call st.stop() here.
    If cooldown is active, just return False and let UI continue rendering.
    """
    now = time.time()
    last = float(st.session_state.get("last_api_call") or 0.0)
    remaining = seconds - (now - last)
    if remaining > 0:
        st.warning(f"Cooldown: wait {remaining:.1f}s and try again.")
        return False
    st.session_state["last_api_call"] = time.time()
    return True

def call_gemini(prompt: str) -> str:
    if not cooldown_ready():
        raise RuntimeError("Cooldown active")
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

def safe_embed_problem(problem: str):
    try:
        if not cooldown_ready():
            return None
        return embed_text(client, problem)
    except Exception:
        return None

def compute_similar_decisions(problem_text: str, threshold: float = 0.45):
    history = load_decisions()
    if not history:
        st.session_state.similar = []
        return

    if not cooldown_ready():
        return

    q_emb = embed_text(client, problem_text)
    top = top_k_similar(q_emb, history, k=3)
    st.session_state.similar = [(s, d) for (s, d) in top if s >= threshold]

def save_future_reflection(timestamp: str, future_text: str) -> bool:
    history = load_decisions()
    updated = False
    for d in history:
        if d.get("timestamp") == timestamp:
            d["future_reflection"] = future_text
            updated = True
            break
    if not updated:
        return False
    rewrite_all_decisions(history)
    return True

# ===================== SIDEBAR (LIGHT) =====================
with st.sidebar:
    logo_path = Path(__file__).resolve().parent / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.caption("Logo file not found: logo.png")

    st.divider()
with st.sidebar:
    st.header("Quick")

    if st.button("Start new decision", key="new_decision"):
        st.session_state.stage = "questions"
        st.session_state.problem = ""
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.priorities = ["Growth"]
        st.session_state.similar = []
        st.session_state.last_result_text = None
        st.session_state.last_saved_timestamp = None
        st.rerun()

    st.divider()
    st.subheader("Similar matches")
    if st.session_state.similar:
        for score, d in st.session_state.similar[:3]:
            st.write(f"**{score:.2f}** — {d.get('problem','')[:70]}…")
    else:
        st.caption("Run “Find similar” in Decide tab.")

# ===================== MAIN TABS =====================
tab_decide, tab_history, tab_insights, tab_settings = st.tabs(
    ["✅ Decide", "📚 History", "📊 Insights", "⚙️ Settings"]
)

# ==========================================================
# TAB: DECIDE
# ==========================================================
with tab_decide:
    step_map = {"questions": 1, "analysis": 2}
    step = step_map.get(st.session_state.stage, 1)
    st.progress(0.25 if step == 1 else 0.60)

    if st.session_state.stage == "questions":
        st.subheader("Step 1 — Describe your decision")
        problem = st.text_area(
            "What decision are you struggling with?",
            value=st.session_state.problem,
            height=160,
            placeholder="Example: Should I change jobs in the next 3 months?"
        )
        st.session_state.problem = problem

        st.markdown("### Actions")

        if st.button("🔎 Find similar decisions", key="find_similar"):
            if not problem.strip():
                st.warning("Please describe your decision first.")
            else:
                with st.spinner("Searching your decision memory..."):
                    try:
                        compute_similar_decisions(problem, threshold=0.45)
                        st.success("Done — see matches in sidebar.")
                    except Exception as e:
                        st.error("Could not compute similar decisions.")
                        st.code(str(e))

        if st.button("➡️ Generate clarifying questions", key="gen_questions"):
            if not problem.strip():
                st.warning("Please describe your decision first.")
            else:
                prompt = questions_prompt(problem, demo_mode=True)
                with st.spinner("Generating questions..."):
                    try:
                        questions_text = call_gemini(prompt)
                    except Exception as e:
                        st.error("Gemini call failed.")
                        st.code(str(e))
                        questions_text = ""

                if questions_text:
                    lines = [l.strip() for l in questions_text.splitlines() if l.strip()]
                    questions = []
                    for line in lines:
                        line = line.lstrip("-• ").strip()
                        if len(line) >= 3 and line[0].isdigit() and line[1] in [".", ")"]:
                            line = line[2:].strip()
                        if len(line) > 3:
                            questions.append(line)

                    if not questions:
                        parts = [p.strip() for p in questions_text.split("?") if p.strip()]
                        questions = [p + "?" for p in parts]

                    questions = questions[:2]
                    if not questions:
                        st.error("Could not parse questions from the model output.")
                    else:
                        st.session_state.questions = questions
                        st.session_state.answers = [""] * len(questions)
                        st.session_state.stage = "analysis"
                        st.rerun()

    elif st.session_state.stage == "analysis":
        st.subheader("Step 2 — Answer a few questions")
        for i, q in enumerate(st.session_state.questions):
            st.session_state.answers[i] = st.text_input(
                f"Q{i+1}: {q}",
                value=st.session_state.answers[i],
                key=f"answer_{i}",
            )

        st.divider()
        st.subheader("Step 3 — Choose priorities")
        st.session_state.priorities = st.multiselect(
            "Pick 1–3 priorities:",
            ["Stability", "Growth", "Money", "Health", "Relationships", "Time freedom", "Confidence"],
            default=st.session_state.priorities or ["Growth"],
        )

        st.divider()
        st.subheader("Step 4 — Get your clarity")
        if st.button("✨ Analyze and show options", key="analyze"):
            if not st.session_state.problem.strip():
                st.warning("Missing problem.")
            elif any(not a.strip() for a in st.session_state.answers):
                st.warning("Please answer all questions.")
            else:
                prompt = analysis_prompt(
                    problem=st.session_state.problem,
                    answers=st.session_state.answers,
                    priorities=st.session_state.priorities,
                    demo_mode=True,
                )

                with st.spinner("Analyzing..."):
                    try:
                        result_text = call_gemini(prompt)
                    except Exception as e:
                        st.error("Gemini call failed.")
                        st.code(str(e))
                        result_text = ""

                if result_text:
                    st.session_state.last_result_text = result_text

                    st.subheader("✅ Clarity result")
                    st.write(result_text)

                    emb = safe_embed_problem(st.session_state.problem)
                    ts = datetime.utcnow().isoformat() + "Z"
                    record = {
                        "timestamp": ts,
                        "problem": st.session_state.problem,
                        "questions": st.session_state.questions,
                        "answers": st.session_state.answers,
                        "priorities": st.session_state.priorities,
                        "result_text": result_text,
                        "embedding": emb,
                        "future_reflection": None,
                    }
                    saved_path = append_decision(record)
                    if saved_path is None:
                        saved_path = Path("decisions.jsonl").resolve()

                    st.session_state.last_saved_timestamp = ts
                    st.success(f"Saved ✅ ({saved_path})")

                    st.divider()
                    st.subheader("🕰️ Perspective shift")
                    if st.button("Ask your future self (6 months later)", key="future_you"):
                        history_now = load_decisions()
                        future_prompt = future_you_prompt(
                            problem=st.session_state.problem,
                            result_text=result_text,
                            past_decisions=history_now,
                            months=6,
                        )
                        with st.spinner("Simulating your future perspective..."):
                            try:
                                future_text = call_gemini(future_prompt)
                            except Exception as e:
                                st.error("Gemini call failed.")
                                st.code(str(e))
                                future_text = ""

                        if future_text:
                            st.markdown("### Future You says:")
                            st.write(future_text)

                            ok = save_future_reflection(ts, future_text)
                            if ok:
                                st.success("Future reflection saved ✅")
                            else:
                                st.warning("Could not attach future reflection (timestamp not found).")

        st.divider()
        if st.button("↩️ Back to Step 1", key="back_step1"):
            st.session_state.stage = "questions"
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.similar = []
            st.rerun()

# ==========================================================
# TAB: HISTORY
# ==========================================================
with tab_history:
    st.subheader("Your saved decisions")
    history = load_decisions()

    if not history:
        st.info("No decisions saved yet.")
    else:
        history_show = list(reversed(history))
        labels = [f"{d.get('timestamp','?')} | {d.get('problem','')[:70]}" for d in history_show]
        pick = st.selectbox("Select:", labels, key="history_pick_main")
        ts = pick.split(" | ", 1)[0].strip()
        st.session_state.history_selected_ts = ts

        d = next((x for x in history_show if x.get("timestamp") == ts), None)
        if not d:
            st.warning("Not found.")
        else:
            st.markdown("### Details")
            st.write(f"**Problem:** {d.get('problem','')}")
            st.write(f"**Priorities:** {', '.join(d.get('priorities') or [])}")

            st.markdown("### Q&A")
            qs = d.get("questions") or []
            ans = d.get("answers") or []
            for i, q in enumerate(qs):
                a = ans[i] if i < len(ans) else ""
                st.write(f"**Q{i+1}:** {q}")
                st.write(f"**A{i+1}:** {a}")

            st.markdown("### Result")
            st.write(d.get("result_text", ""))

            if d.get("future_reflection"):
                st.markdown("### Future reflection")
                st.write(d.get("future_reflection"))

            st.divider()
            st.markdown("### Actions")
            confirm = st.checkbox("I understand this deletes permanently.", key="confirm_delete_main")
            if st.button("🗑 Delete this decision", key="delete_decision_main"):
                if not confirm:
                    st.warning("Please confirm first.")
                else:
                    ok = delete_decision_by_timestamp(ts)
                    if ok:
                        st.success("Deleted ✅")
                        st.session_state.similar = []
                        st.rerun()
                    else:
                        st.warning("Could not delete.")

# ==========================================================
# TAB: INSIGHTS
# ==========================================================
with tab_insights:
    st.subheader("Insights")
    history = load_decisions()
    if not history:
        st.info("Add a few decisions to unlock insights.")
    else:
        try:
            ins = compute_insights(history)
        except Exception as e:
            st.error("Insights failed (non-critical).")
            st.code(str(e))
            ins = {"total": len(history), "priority_counts": {}, "topic_counts": {}, "habit": ""}

        c1, c2, c3 = st.columns(3)
        c1.metric("Total decisions", ins.get("total", 0))

        top_pr = format_top(ins.get("priority_counts", {}), k=3)
        top_topics = format_top(ins.get("topic_counts", {}), k=3)

        with c2:
            st.write("**Top priorities**")
            for name, cnt in top_pr:
                st.write(f"- {name}: {cnt}")

        with c3:
            st.write("**Top areas**")
            for name, cnt in top_topics:
                st.write(f"- {name}: {cnt}")

        st.divider()
        st.write("**Next best habit**")
        st.info(ins.get("habit", "Keep adding decisions to improve insights."))

# ==========================================================
# TAB: SETTINGS
# ==========================================================
with tab_settings:
    st.subheader("Settings")
    st.caption("User settings are minimal by design to keep the experience clean.")

    st.write("**Rate limiting**")
    st.caption("Cooldown prevents accidental double-click spam of the Gemini API.")
    st.write(f"Current cooldown: **{COOLDOWN_SECONDS:.1f}s**")

    st.divider()

    if ADMIN_MODE:
        st.subheader("Admin tools")
        st.caption("Visible only when CLARITY_ADMIN=1")

        with st.expander("Maintenance"):
            if st.button("Rebuild embeddings (ALL decisions)", key="admin_rebuild_embeddings"):
                history = load_decisions()
                if not history:
                    st.info("No decisions found.")
                else:
                    with st.spinner("Rebuilding embeddings..."):
                        updated = []
                        for d in history:
                            problem = (d.get("problem") or "").strip()
                            if not problem:
                                d["embedding"] = None
                                updated.append(d)
                                continue

                            if not cooldown_ready():
                                st.info("Cooldown active — click again in a moment.")
                                updated = None
                                break

                            d["embedding"] = embed_text(client, problem)
                            updated.append(d)

                        if updated is not None:
                            saved_path = rewrite_all_decisions(updated)
                            st.success(f"Rebuilt ✅ ({saved_path})")
                            st.session_state.similar = []
                            st.rerun()

            if st.button("Reset cooldown", key="admin_reset_cooldown"):
                st.session_state["last_api_call"] = 0.0
                st.success("Reset ✅")

        if ENABLE_ML_TOPICS_UI and TOPICS_ML_AVAILABLE:
            with st.expander("ML Topics (Gemini embeddings)"):
                st.session_state["use_ml_topics"] = st.checkbox(
                    "Enable ML Topics (admin)",
                    value=st.session_state.get("use_ml_topics", False),
                )
                if st.session_state["use_ml_topics"]:
                    if st.session_state.get("topic_centroids") is None:
                        st.caption("Centroids not built yet. Build once (uses embedding API).")
                        if st.button("Build topic centroids (one-time)", key="admin_build_centroids"):
                            with st.spinner("Building centroid embeddings..."):
                                if not cooldown_ready():
                                    st.info("Cooldown active — click again in a moment.")
                                else:
                                    st.session_state["topic_centroids"] = build_topic_centroids(client, embed_text)
                                    st.success("Centroids built ✅")
                    else:
                        st.caption("Centroids ready ✅")
    else:
        st.caption("Admin tools are disabled in this build.")