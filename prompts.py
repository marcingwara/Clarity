import re

def detect_language(text: str) -> str:
    """
    Wykrywa język tekstu i zwraca kod ISO.
    """
    text_lower = text.lower()

    # Charakterystyczne słowa - sprawdzamy z spacjami by uniknąć false positives
    polish_indicators = ['czy', 'jak', 'mam', 'jest', 'nie', 'tak', 'ale', 'się', 'że', 'będę', 'chcę', 'mój', 'moja', 'powinien']
    english_indicators = ['the', 'is', 'am', 'are', 'should', 'can', 'have', 'what', 'how', 'when', 'my', 'need', 'want', 'whether']
    spanish_indicators = ['que', 'como', 'tengo', 'debo', 'puedo', 'hay', 'para', 'con', 'mi', 'necesito', 'debería']
    french_indicators = ['que', 'comme', 'dois', 'peux', 'suis', 'pour', 'avec', 'mon', 'besoin', 'devrais']

    # Zliczanie ze spacjami (bardziej dokładne)
    pl_count = sum(1 for word in polish_indicators if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} '))
    en_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} '))
    es_count = sum(1 for word in spanish_indicators if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} '))
    fr_count = sum(1 for word in french_indicators if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} '))

    scores = {'pl': pl_count, 'en': en_count, 'es': es_count, 'fr': fr_count}
    detected = max(scores, key=scores.get)

    return detected if scores[detected] > 0 else 'en'


def get_system_instruction(lang_code: str) -> str:
    """
    Zwraca system instruction dla Gemini API - WYMUSZA język odpowiedzi.
    """
    instructions = {
        'pl': "Odpowiadaj WYŁĄCZNIE po polsku. Wszystkie nagłówki, opisy, liczby, tabele i tekst muszą być po polsku. Nie używaj żadnych innych języków.",
        'en': "Respond EXCLUSIVELY in English. All headings, descriptions, numbers, tables and text must be in English. Do not use any other languages.",
        'es': "Responde EXCLUSIVAMENTE en español. Todos los encabezados, descripciones, números, tablas y texto deben estar en español. No uses ningún otro idioma.",
        'fr': "Répondez EXCLUSIVEMENT en français. Tous les en-têtes, descriptions, nombres, tableaux et texte doivent être en français. N'utilisez aucune autre langue."
    }
    return instructions.get(lang_code, instructions['en'])


def questions_prompt(problem: str, demo_mode: bool) -> str:
    """
    Prompt dla pytań - prosty, bez instrukcji językowych.
    """
    return f"""You are Clarity — an AI companion for thoughtful decision-making.

USER PROBLEM:
{problem}

YOUR TASK:
Ask exactly {2 if demo_mode else 3} short, practical clarifying questions.
Format: numbered list (1), 2), 3)).
Do NOT give advice or analyze options.
Do NOT add any other text."""


def analysis_prompt(
        problem: str,
        answers: list[str],
        priorities: list[str],
        demo_mode: bool
) -> str:
    """
    Prompt analizy - prosty, bez instrukcji językowych.
    """
    priorities_text = ", ".join(priorities) if priorities else "Not specified"
    answers_text = "\n".join([f"{i+1}) {answers[i]}" for i in range(len(answers))])

    return f"""You are Clarity — a calm, practical decision mentor.

DECISION PROBLEM:
{problem}

USER ANSWERS:
{answers_text}

USER PRIORITIES:
{priorities_text}

FORMATTING RULES:
- Use headings with ### and make section titles bold with **
- Section titles must be in LARGER font and BOLD
- For section 5 "Pros vs Cons comparison", use a table format
- Numbers must be explicit and consistent

Produce the response EXACTLY in this structure:

### **1) Clarify constraints**
- **Constraint 1:** concrete limitation (money / time / obligation)
- **Constraint 2:** concrete limitation
- **Constraint 3:** concrete limitation
- **Missing information:** explicitly state what critical data is missing (if any)

### **2) Restate the decision**
- **One clear sentence** describing what the user is deciding.

### **3) Possible options**

#### **Option 1**
- **Description:** short and concrete
- **Immediate gains:** what improves right away
- **Immediate losses:** what is sacrificed

#### **Option 2**
- **Description:** short and concrete
- **Immediate gains:** what improves right away
- **Immediate losses:** what is sacrificed

### **4) Numerical evaluation**

#### **Option 1**
- **Overall fit:** XX / 100
- **Confidence level:** XX %
- **Reasoning:** 1–2 concrete sentences tied to priorities and constraints

#### **Option 2**
- **Overall fit:** XX / 100
- **Confidence level:** XX %
- **Reasoning:** 1–2 concrete sentences tied to priorities and constraints

### **5) Pros vs Cons comparison (percentage-based)**

| Option | Pros Strength | Cons Weight | Summary |
|--------|---------------|-------------|---------|
| **Option 1** | XX% | XX% | [one practical sentence explaining the balance] |
| **Option 2** | XX% | XX% | [one practical sentence explaining the balance] |

*(Pros % + Cons % MUST equal 100 for each option)*

### **6) Clear recommendation**
- **Best option RIGHT NOW:** explicitly state Option 1 or Option 2
- **Why:** concrete reasoning, no motivation fluff
- **This recommendation would change if:** one clear condition

### **7) Short-term action plan (next 7 days)**
- **Action 1:** measurable, with number or deadline
- **Action 2:** measurable
- **Action 3:** measurable

### **8) Safety rules**
- **Backup trigger:** one concrete condition
- **Overload rule:** one simple behavioral rule
- **Review point:** specific date or condition

Tone: calm, grounded, practical, zero motivational fluff."""


def future_you_prompt(problem: str, result_text: str, past_decisions: list, months: int = 6) -> str:
    """
    Prompt dla "future you".
    """
    memory = []
    for d in past_decisions[-3:]:
        memory.append(
            f"- Problem: {d.get('problem','')}\n"
            f"  Priorities: {', '.join(d.get('priorities') or [])}\n"
            f"  Result: {d.get('result_text','')[:180]}"
        )
    memory_block = "\n".join(memory) if memory else "No past decisions."

    return f"""You are the SAME person, but {months} months in the future.

USER PROBLEM:
{problem}

ORIGINAL ANALYSIS (what you believed then):
{result_text}

PAST DECISIONS FROM YOUR LIFE (context):
{memory_block}

TASK:
Write a grounded reflection from the future perspective:
1) What turned out to matter most?
2) What I underestimated at the time
3) One short advice sentence for my past self

Rules:
- Supportive but honest tone
- No generic motivational fluff
- 120–180 words"""


# ============================================================
# FUNKCJA DO DODANIA W app.py - ZASTĄP call_gemini()
# ============================================================

def call_gemini_with_language(client, prompt: str, problem_text: str) -> str:
    """
    Wywołuje Gemini API z automatyczną detekcją języka.

    Args:
        client: Gemini client
        prompt: Wygenerowany prompt (questions_prompt, analysis_prompt, etc.)
        problem_text: Oryginalny tekst problemu użytkownika (do detekcji języka)

    Returns:
        str: Odpowiedź od Gemini w języku użytkownika
    """
    from google import genai

    # Wykryj język z problemu użytkownika
    lang_code = detect_language(problem_text)

    # Pobierz system instruction dla tego języka
    system_inst = get_system_instruction(lang_code)

    # KLUCZOWE: Użyj GenerativeModel z system_instruction
    model = genai.GenerativeModel(
        model_name="models/gemini-flash-lite-latest",
        system_instruction=system_inst  # ← TO WYMUSZA JĘZYK!
    )

    # Wywołaj API
    response = model.generate_content(prompt)

    return response.text if hasattr(response, "text") else str(response)