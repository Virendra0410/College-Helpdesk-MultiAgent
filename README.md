
# 🎓 College Helpdesk Multi-Agent System

A multi-agent AI-powered college helpdesk that answers student questions about examinations, fees, admissions, academic calendar, and attendance.

## 🚀 Features

- Multi-agent architecture using LangGraph
- Conditional question routing
- Specialized agents for:
  - Examination
  - Fees
  - Admission
  - Academic Calendar
  - General / Attendance
- Retrieval-Augmented Generation (RAG)
- ChromaDB vector stores
- Hugging Face embeddings
- Pydantic response validation
- Conversation history
- Streamlit web interface
- College-specific PDF documents as the knowledge source

URL - https://college-appdesk-multiagent-fymmz3w6vg98tvs7ybsfkv.streamlit.app/

## 🏗️ Architecture

```text
Student
   ↓
Streamlit Interface
   ↓
Router Agent
   ↓
┌────────────┬────────────┬──────────────┬────────────┬─────────────┐
│ Exam Agent │ Fee Agent  │ Admission    │ Calendar   │ General     │
│            │            │ Agent        │ Agent      │ Agent       │
└────────────┴────────────┴──────────────┴────────────┴─────────────┘
                         ↓
                  Structured RAG
                         ↓
                     ChromaDB
                         ↓
                 Pydantic Validation
                         ↓
                   Final Response



