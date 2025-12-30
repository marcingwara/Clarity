# 🧠 Clarity – Better decisions, built over time

Clarity is a decision-support app that helps you **think better before you decide**  
and **learn from your past decisions afterward**.

Instead of giving answers, Clarity asks the *right questions*, surfaces *patterns* in your life,
and lets your **future self** reflect on today’s choices.

---

## ✨ What problem does it solve?

People don’t fail because they lack information.  
They fail because they:
- repeat the same decision patterns
- ignore their own priorities
- don’t learn from past choices

Clarity helps you:
- slow down impulsive decisions
- clarify what really matters
- reuse your own decision history as intelligence

---

## 🚀 Key features

### 1️⃣ Clarifying questions (Gemini-powered)
You describe a decision, and Clarity generates **focused, non-generic questions**
to help you think before acting.

### 2️⃣ Structured analysis
Based on your answers and priorities, Clarity produces a grounded analysis:
- trade-offs
- risks
- realistic options (no motivational fluff)

### 3️⃣ Decision memory (local, private)
Each decision is saved with:
- problem
- questions & answers
- priorities
- analysis
- embedding (vector representation)

This creates a **personal decision dataset** over time.

### 4️⃣ Similar past decisions (ML)
Using **Gemini embeddings**, Clarity finds decisions from your past
that are semantically similar — even if phrased differently.

> “You’ve been here before. This is what you chose last time.”

### 5️⃣ Insights & patterns (beta)
Clarity detects:
- repeated priorities by topic
- dominant decision styles (Explorer, Stabilizer, etc.)
- gentle habit suggestions based on your behavior

### 6️⃣ 🕰️ “Future You” reflection
After analysis, you can ask:
> *“What would I think about this decision 6 months from now?”*

Gemini simulates a **future perspective**, grounded in your real past decisions.

---

## 🧠 Why Gemini?

Clarity uses **Gemini exclusively** for:
- text generation (questions, analysis, future reflection)
- embeddings (`text-embedding-004`) for semantic memory & similarity

Why Gemini fits this project:
- strong reasoning for reflective prompts
- high-quality embeddings for meaning (not keywords)
- single ecosystem (no external ML dependencies)

This makes Clarity:
- simple to deploy
- transparent
- competition-friendly

---

## 🛠️ Tech stack

- **Python**
- **Streamlit** – UI
- **Google Gemini API**
- **NumPy** – similarity math
- **JSONL** – lightweight local storage

No databases. No external ML frameworks.  
Just your decisions + Gemini.

---

## 📂 Project structure

```text
Clarity/
├── app.py              # Streamlit app
├── insights.py         # Pattern & behavior analysis
├── ml.py               # Gemini embeddings + similarity
├── memory.py           # Decision storage (JSONL)
├── prompts.py          # Prompt templates
├── decisions.jsonl     # Local decision memory
├── .env                # API key (ignored)
└── README.md

```
HowHow to run locally
```
pip install -r requirements.txt
```
Create .env:
```
GEMINI_API_KEY=your_api_key_here
```
Run:
```
streamlit run app.py
```
## 🔒 Privacy & data
- All decisions are stored locally
- No user data is sent anywhere except to Gemini for inference
- No analytics, no tracking

Your decisions stay yours.

## 🧪 Project status
- Core features: ✅ done
- Insights & patterns: 🧪 beta
- UI polish: 🚧 ongoing
- Long-term vision: personal decision intelligence

## 🧭 Future ideas
- timeline view of decisions
- visual decision patterns
- long-term outcome tracking
- optional cloud sync (opt-in)

## 👤 Author
Marcin Gwara
