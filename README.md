# 🎓 College Helpdesk Multi-Agent System

A genuinely LLM-powered, multi-agent Retrieval-Augmented Generation (RAG) college helpdesk that dynamically answers student questions about examinations, fees, admissions, academic calendar, and attendance rules using official college documents.

Deployed on Streamlit Community Cloud:  
🔗 **[Live Application](https://college-appdesk-multiagent-fymmz3w6vg98tvs7ybsfkv.streamlit.app/)**

---

## 🚀 Key Features

- **LLM-Powered Multi-Agent Architecture**: Built with LangGraph conditional routing and specialized agents.
- **Dynamic Grounded RAG**: Official college PDFs are embedded using Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`) and queried via ChromaDB vector stores.
- **Google Gemini Flash Integration**: Uses Google Gemini Flash (configurable via environment variables) for question classification and factual answer synthesis.
- **Strict Anti-Hallucination & Fee Safeguards**: Specialized prompts and document metadata prevent fee confusion (e.g. asking for B.Tech fees will never return M.Tech fees; missing programme information is explicitly stated as unavailable).
- **Session-Scoped Conversation Memory**: Multi-turn history is preserved within each session to naturally resolve follow-up questions (e.g., "What about the winter semester?").
- **Pydantic Response Validation**: Ensures structured outputs, category validation, and citation of source documents.
- **Streamlit Chat Interface**: Clean user interface featuring category attribution, source document tags, loading spinners, and secure error handling.

---

## 🏗️ Architecture

```text
                           Student
                              ↓
                   Streamlit Chat Interface
                              ↓
                      LangGraph Workflow
                              ↓
                    LLM-Powered Router
                              ↓
    ┌────────────┬────────────┬──────────────┬────────────┬─────────────┐
    │ Exam Agent │ Fee Agent  │  Admission   │  Calendar  │   General   │
    │            │            │    Agent     │   Agent    │    Agent    │
    └────────────┴────────────┴──────────────┴────────────┴─────────────┘
                              ↓
                Relevant Chroma Retriever (Top-K)
                              ↓
                   Retrieved Document Chunks
                              ↓
               Google Gemini Flash (Grounded RAG)
                              ↓
                  Pydantic Response Validation
                              ↓
                  Final Response + Citations
```

---

## 📚 Knowledge Base Documents

The system grounds all answers in the official NIT Calicut documents located in `documents/`:
1. `fee_structure.pdf`: Academic fees for M.Tech/M.Plan, M.Sc., MBA, and Ph.D.
2. `exam_malpractice_guidelines.pdf`: Guidelines on penalties, malpractice types, and disciplinary actions.
3. `academic_calendar.pdf`: Key academic dates for the Monsoon Semester 2026-27.
4. `admission_rules.pdf`: Physical reporting venue, schedules by department, and required original documents.
5. `student_handbook.pdf`: Student ordinances, attendance requirements, shortage notifications, and condonation policies.

---

## ⚙️ Configuration & Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API Key from Google AI Studio | **Required** |
| `GEMINI_MODEL` | Gemini Flash model identifier | `gemini-3.5-flash-lite` |

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Virendra0410/College-Helpdesk-MultiAgent.git
   cd CollegeHelpdeskAgent
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API Key**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Add your Gemini API key in `.env`:
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment on Streamlit Community Cloud

When deploying or updating on Streamlit Community Cloud:

1. Open your app dashboard on [share.streamlit.io](https://share.streamlit.io).
2. Go to **Settings** > **Secrets**.
3. Add your secrets in TOML format:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key"
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   ```
4. Save the secrets and restart/redeploy the app.

---

## 🧪 Testing and Evaluation

Run the automated test suite covering routing, Pydantic validation, anti-hallucination, and all 10 standard evaluation questions:

```bash
pytest tests/test_helpdesk.py -v
```

### Verified Test Cases:
1. Impersonation in examinations (`exam_agent` -> F grade & 2 semesters debarment)
2. M.Tech tuition fees (`fee_agent` -> ₹35,000 per listed semester)
3. B.Tech fees anti-hallucination check (`fee_agent` -> confirms B.Tech fee is absent, never outputs ₹35,000)
4. Physical reporting venue (`admission_agent` -> Bhaskara Hall, NIT Calicut)
5. Semester commencement date (`calendar_agent` -> 21 July 2026)
6. Attendance below 80% (`general_agent` -> shortage notice & 50% condonation rule)
7. Add/Drop deadline (`calendar_agent` -> 30 July 2026)
8. Admission documents required (`admission_agent` -> Class X DOB proof, original certificates)
9. Missing/out-of-scope information handling (Graceful "could not find" response)
10. Multi-turn follow-up resolution (Context-aware disambiguation of follow-up questions)
