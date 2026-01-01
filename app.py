import os
import time
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import streamlit as st
from dotenv import load_dotenv
from google import genai

from prompts import questions_prompt, analysis_prompt, future_you_prompt
from memory import load_decisions, append_decision, delete_decision_by_timestamp, rewrite_all_decisions
from ml import embed_text, top_k_similar
from insights import compute_insights, format_top
from topics_ml import build_topic_centroids, classify_from_embedding

# ===================== BASIC SETUP =====================
load_dotenv()
st.set_page_config(page_title="Clarity", page_icon="🧠", layout="centered")

def get_api_key() -> str | None:
    # Streamlit Cloud: prefer st.secrets, local: .env / env var
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None

api_key = get_api_key()
if not api_key:
    st.error("Missing GEMINI_API_KEY. Add it to Streamlit Secrets (or .env locally).")
    st.stop()

client = genai.Client(api_key=api_key)
CHAT_MODEL = "models/gemini-flash-lite-latest"

# ===================== SESSION STATE =====================
def init_state():
    defaults = {
        "stage": "questions",
        "problem": "",
        "questions": [],
        "answers": [],
        "similar": [],
        "last_api_call": 0.0,
        "history_selected": None,
        "last_saved_timestamp": None,
        "topic_centroids": None,
        "use_ml_topics": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ===================== COOLDOWN =====================
COOLDOWN_SECONDS = 1.0

def wait_for_cooldown(seconds: float = COOLDOWN_SECONDS) -> None:
    now = time.time()
    last = float(st.session_state.get("last_api_call") or 0.0)
    remaining = seconds - (now - last)
    if remaining > 0:
        st.info(f"Cooldown: wait {remaining:.1f}s and click again.")
        st.stop()
    st.session_state["last_api_call"] = time.time()

def call_gemini(prompt: str) -> str:
    wait_for_cooldown()
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

# ===================== HELPERS =====================
def compute_similar_decisions(problem_text: str, threshold: float = 0.45) -> None:
    history = load_decisions()
    if not history:
        st.session_state.similar = []
        return

    wait_for_cooldown()
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

def safe_compute_insights(history):
    try:
        return compute_insights(history)
    except Exception as e:
        return {"total": len(history), "error": str(e)}

def safe_ml_topics_overrides(history, ins, centroids):
    """
    Override topic_counts + pattern_rules using ML topics,
    but ONLY when embeddings exist (no API calls).
    """
    topics = []
    for d in history:
        emb = d.get("embedding")
        if not emb:
            topics.append("unknown")
            continue
        topic, _score = classify_from_embedding(emb, centroids, min_score=0.35)
        topics.append(topic)

    topic_counts = Counter(topics)

    pr_by_topic = defaultdict(Counter)
    for d, t in zip(history, topics):
        for p in (d.get("priorities") or []):
            pr_by_topic[t][p] += 1

    pattern_rules = []
    for t, c in pr_by_topic.items():
        if not c:
            continue
        top_p, top_cnt = c.most_common(1)[0]
        if top_cnt >= 2 or len(history) <= 3:
            pattern_rules.append({"topic": t, "top_priority": top_p, "count": top_cnt})

    ins["topic_counts"] = topic_counts
    ins["pattern_rules"] = pattern_rules
    return ins

# ===================== UI =====================
st.title("🧠 Clarity")
st.caption("Ask first. Understand second. Decide last.")

demo_mode = st.toggle("🎥 Demo mode (shorter output)", value=True, key="toggle_demo")
debug = st.toggle("🛠 Debug mode", value=False, key="toggle_debug")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("🧠 Decision Memory")
    st.caption(f"Working directory: {os.getcwd()}")

    # -------- Maintenance --------
    st.subheader("🧹 Maintenance")

    if st.button("Rebuild embeddings (ALL decisions)", key="btn_rebuild_embeddings"):
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
                    try:
                        wait_for_cooldown()
                        d["embedding"] = embed_text(client, problem)
                    except Exception as e:
                        d["embedding"] = None
                        if debug:
                            st.write(f"Embedding failed for: {problem[:40]}")
                            st.code(str(e))
                    updated.append(d)

                saved_path = rewrite_all_decisions(updated)
                st.success(f"Rebuilt embeddings ✅ ({saved_path})")
                st.session_state.similar = []
                st.rerun()

    if st.button("Reset cooldown (dev)", key="btn_reset_cooldown"):
        st.session_state["last_api_call"] = 0.0
        st.success("Cooldown reset.")

    st.divider()

    # -------- History Browser --------
    st.subheader("📚 History Browser")
    history = load_decisions()

    if not history:
        st.caption("No saved decisions yet.")
        selected = None
    else:
        history_show = list(reversed(history[-20:]))
        labels = [f"{d.get('timestamp','?')} | {d.get('problem','')[:50]}" for d in history_show]

        selected_label = st.selectbox("Pick a saved decision:", labels, key="sb_history_pick")
        selected_ts = selected_label.split(" | ", 1)[0].strip()
        selected = next((d for d in history_show if d.get("timestamp") == selected_ts), None)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Open", key="btn_open_details"):
                st.session_state["history_selected"] = selected
        with col2:
            confirm = st.checkbox("Confirm delete", key="chk_confirm_delete_sidebar")
            if st.button("🗑 Delete", key="btn_delete_sidebar"):
                if not confirm:
                    st.warning("Tick Confirm delete first.")
                else:
                    ts = (selected or {}).get("timestamp")
                    if ts and delete_decision_by_timestamp(ts):
                        st.success("Deleted ✅")
                        st.session_state["history_selected"] = None
                        st.session_state.similar = []
                        st.rerun()
                    else:
                        st.warning("Could not delete.")

    st.divider()

    # -------- ML Topics Controls --------
    st.subheader("🧠 ML Topics (Gemini embeddings)")
    st.session_state["use_ml_topics"] = st.checkbox(
        "Use ML topics in Insights",
        value=bool(st.session_state.get("use_ml_topics")),
        key="chk_use_ml_topics"
    )

    if st.session_state["use_ml_topics"]:
        if st.session_state.get("topic_centroids") is None:
            st.caption("Centroids not built yet. Build once (uses embedding API).")
            if st.button("Build topic centroids (one-time)", key="btn_build_centroids"):
                with st.spinner("Building centroid embeddings..."):
                    try:
                        wait_for_cooldown()
                        st.session_state["topic_centroids"] = build_topic_centroids(client, embed_text)
                        st.success("Centroids built ✅")
                    except Exception as e:
                        st.session_state["topic_centroids"] = None
                        st.error("Failed to build centroids.")
                        if debug:
                            st.code(str(e))
        else:
            st.caption("Centroids ready ✅")

    st.divider()

    # -------- Insights --------
    st.subheader("📊 Insights")
    if not history:
        st.caption("Add a few decisions to unlock insights.")
    else:
        ins = safe_compute_insights(history)

        # optional ML topics override
        if st.session_state["use_ml_topics"] and st.session_state.get("topic_centroids"):
            ins = safe_ml_topics_overrides(history, ins, st.session_state["topic_centroids"])

        if ins.get("error"):
            st.warning("Insights had a non-critical error.")
            if debug:
                st.code(ins["error"])

        st.metric("Total decisions", ins.get("total", 0))

        top_pr = format_top(ins.get("priority_counts", Counter()), k=3)
        if top_pr:
            st.write("**Top priorities**")
            for name, cnt in top_pr:
                st.write(f"- {name}: {cnt}")

        top_topics = format_top(ins.get("topic_counts", Counter()), k=3)
        if top_topics:
            st.write("**Top decision areas**")
            for name, cnt in top_topics:
                st.write(f"- {name}: {cnt}")

        if ins.get("top_words"):
            st.write("**Common themes (words)**")
            st.caption(", ".join([w for w, _ in ins["top_words"]]))

        st.divider()
        st.subheader("🧩 Patterns (beta)")
        patterns = ins.get("pattern_rules") or []
        if patterns:
            for r in patterns[:5]:
                st.write(f"- {r.get('topic','?')}: usually you pick **{r.get('top_priority','?')}** ({r.get('count',0)}x)")
        else:
            st.caption("Not enough data yet for patterns.")

        style_counts = ins.get("style_counts")
        if hasattr(style_counts, "most_common"):
            top_style = style_counts.most_common(1)
            if top_style:
                st.write(f"**Decision style:** {top_style[0][0]}")

        sig = ins.get("style_signals", {})
        if isinstance(sig, dict) and sig:
            st.caption(
                f"Signals - risk:{sig.get('risk_focus_ratio', 0)}, "
                f"action:{sig.get('action_focus_ratio', 0)}, "
                f"overthink:{sig.get('overthink_ratio', 0)}"
            )

        habit = ins.get("habit")
        if habit:
            st.write("**Next best habit**")
            st.info(habit)

    st.divider()

    # -------- Similar decisions --------
    st.subheader("🔎 Similar decisions")
    if st.session_state.similar:
        for score, d in st.session_state.similar:
            st.write(f"**Similarity:** {score:.2f}")
            st.write(f"**When:** {d.get('timestamp', 'unknown')}")
            st.write(f"**Problem:** {d.get('problem', '')[:160]}...")
            st.divider()
    else:
        st.caption("No similar decisions yet (or similarity below threshold).")

st.divider()

# ===================== HISTORY DETAILS (MAIN AREA) =====================
if st.session_state.get("history_selected"):
    d = st.session_state["history_selected"]

    st.subheader("📄 Saved decision details")
    st.write(f"**Timestamp:** {d.get('timestamp','?')}")
    st.write(f"**Problem:** {d.get('problem','')}")
    st.write(f"**Priorities:** {', '.join(d.get('priorities') or [])}")

    st.markdown("### Q&A")
    qs = d.get("questions") or []
    ans = d.get("answers") or []
    for i, q in enumerate(qs):
        a = ans[i] if i < len(ans) else ""
        st.write(f"**Q{i+1}:** {q}")
        st.write(f"**A{i+1}:** {a}")
        st.divider()

    st.markdown("### Result")
    st.write(d.get("result_text", ""))

    if d.get("future_reflection"):
        st.markdown("### 🕰️ Future reflection")
        st.write(d.get("future_reflection"))

    if st.button("Close details", key="btn_close_details"):
        st.session_state["history_selected"] = None
        st.rerun()

st.divider()

# ===================== STAGE 1 =====================
if st.session_state.stage == "questions":
    st.subheader("1) Describe your decision")

    problem = st.text_area(
        "What decision are you struggling with?",
        value=st.session_state.problem,
        height=120,
        key="ta_problem",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🔎 Find similar decisions (ML)", key="btn_find_similar"):
            if not problem.strip():
                st.warning("Please describe your decision first.")
            else:
                st.session_state.problem = problem
                with st.spinner("Searching your decision memory..."):
                    try:
                        compute_similar_decisions(problem, threshold=0.45)
                    except Exception as e:
                        st.error("Could not compute similar decisions.")
                        if debug:
                            st.code(str(e))

    with col_b:
        if st.button("2) Generate clarifying questions", key="btn_gen_questions"):
            if not problem.strip():
                st.warning("Please describe your decision first.")
            else:
                st.session_state.problem = problem
                prompt = questions_prompt(problem, demo_mode)

                if debug:
                    st.subheader("Prompt (questions)")
                    st.code(prompt)

                with st.spinner("Generating questions..."):
                    try:
                        questions_text = call_gemini(prompt)
                    except Exception as e:
                        st.error("Gemini call failed while generating questions.")
                        if debug:
                            st.code(str(e))
                        st.stop()

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

                questions = questions[: (2 if demo_mode else 3)]
                if not questions:
                    st.error("Could not parse questions from the model output.")
                    st.stop()

                st.session_state.questions = questions
                st.session_state.answers = [""] * len(questions)
                st.session_state.stage = "analysis"
                st.rerun()

# ===================== STAGE 2 =====================
if st.session_state.stage == "analysis":
    st.subheader("2) Answer the questions")

    for i, question in enumerate(st.session_state.questions):
        st.session_state.answers[i] = st.text_input(
            f"Question {i + 1}: {question}",
            value=st.session_state.answers[i] if i < len(st.session_state.answers) else "",
            key=f"answer_{i}",
        )

    st.divider()
    st.subheader("3) What matters most to you?")

    priorities = st.multiselect(
        "Choose 1–3 priorities:",
        ["Stability", "Growth", "Money", "Health", "Relationships", "Time freedom", "Confidence"],
        default=["Growth"],
        key="ms_priorities",
    )

    st.divider()

    if st.button("4) Analyze and show options", key="btn_analyze"):
        if any(not a.strip() for a in st.session_state.answers):
            st.warning("Please answer all questions.")
            st.stop()

        prompt = analysis_prompt(
            problem=st.session_state.problem,
            answers=st.session_state.answers,
            priorities=priorities,
            demo_mode=demo_mode,
        )

        if debug:
            st.subheader("Prompt (analysis)")
            st.code(prompt)

        with st.spinner("Analyzing..."):
            result_text = call_gemini(prompt)

        st.subheader("✅ Clarity result")
        st.write(result_text)

        # ---- Embedding (optional) ----
        emb = None
        try:
            wait_for_cooldown()
            emb = embed_text(client, st.session_state.problem)
        except Exception as e:
            emb = None
            if debug:
                st.warning("Embedding failed (non-critical).")
                st.code(str(e))

        # ---- Save decision FIRST ----
        ts = datetime.utcnow().isoformat() + "Z"
        record = {
            "timestamp": ts,
            "problem": st.session_state.problem,
            "questions": st.session_state.questions,
            "answers": st.session_state.answers,
            "priorities": priorities,
            "result_text": result_text,
            "embedding": emb,
            "future_reflection": None,
        }

        saved_path = append_decision(record)
        if saved_path is None:
            saved_path = Path("decisions.jsonl").resolve()
        st.success(f"Saved to Decision Memory ✅ ({saved_path})")

        st.session_state["last_saved_timestamp"] = ts

        # ---- Future You (on demand) ----
        st.divider()
        st.subheader("🕰️ Perspective shift")

        if st.button("Ask your future self (6 months later)", key="btn_future_you"):
            history_now = load_decisions()
            with st.spinner("Simulating your future perspective..."):
                future_prompt = future_you_prompt(
                    problem=st.session_state.problem,
                    result_text=result_text,
                    past_decisions=history_now,
                    months=6,
                )

                if debug:
                    st.subheader("Prompt (future)")
                    st.code(future_prompt)

                future_text = call_gemini(future_prompt)

            st.markdown("### 🧠 Future You says:")
            st.write(future_text)

            ok = save_future_reflection(ts, future_text)
            if ok:
                st.success("Future reflection saved ✅")
            else:
                st.warning("Could not attach future reflection to the saved decision (timestamp not found).")

    st.divider()

    if st.button("↩️ Start over", key="btn_start_over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()