"""
College Helpdesk Multi-Agent Backend
Powered by LangChain, LangGraph, ChromaDB, Hugging Face Embeddings,
and Google Gemini Flash.
"""

import os
import re
import logging
from typing import TypedDict, Literal, Optional, List, Tuple, Any

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

from llm import invoke_llm_with_fallback, get_api_key, clean_response_text

logger = logging.getLogger(__name__)

DOCUMENTS_PATH = "documents"
CHROMA_PERSIST_DIR = "chroma_db"

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# STRUCTURED CORE COLLEGE DOCUMENTS
# --------------------------------------------------

structured_exam_documents = [
    Document(
        page_content="""
ITEM 11 — IMPERSONATION IN THE EXAMINATION

Nature of malpractice:
Impersonation in the examination (for both the students).

Punishment:
The concerned students shall be awarded 'F' grade in all subjects credited in the corresponding semester.
AND
The concerned students shall be debarred from attending classes and appearing for examinations in the next two subsequent semesters.
""",
        metadata={
            "source": "exam_malpractice_guidelines.pdf",
            "item": 11,
            "category": "examination malpractice"
        }
    ),
    Document(
        page_content="""
ITEM 12 — MALPRACTICE DURING REX/SAY/SUPPLEMENTARY EXAMINATIONS

Nature of malpractice:
Students engaged in malpractice during REX/SAY/Supplementary examinations.

Punishment:
The concerned student(s) shall be awarded 'F' grade in the corresponding course.
In addition, the student(s) shall be awarded one grade less for all the other theory courses registered in the corresponding REX/SAY/Supplementary examinations.
AND
The student(s) shall be debarred from the provision of any REX/SAY/Supplementary examinations in future.
""",
        metadata={
            "source": "exam_malpractice_guidelines.pdf",
            "item": 12,
            "category": "examination malpractice"
        }
    ),
    Document(
        page_content="""
ITEM 13 — ANY OTHER CASE NOT MENTIONED ABOVE

Nature of malpractice:
Any other case not mentioned above.

Punishment:
Punishment will be as recommended by the Department level committee for malpractice enquiry or Institute level committee for malpractice enquiry.
""",
        metadata={
            "source": "exam_malpractice_guidelines.pdf",
            "item": 13,
            "category": "examination malpractice"
        }
    )
]

structured_fee_documents = [
    Document(
        page_content="""
NIT CALICUT — ACADEMIC FEES FOR 2026-27 ADMISSION BATCH

Programme: M.Tech / M.Plan
Category:
1. Students admitted through CCMT
2. All students admitted through QIP

Semester-wise fees:
Tuition Fee:
- Monsoon Semester 2026-27: ₹35,000
- Winter Semester 2026-27: ₹35,000
- Monsoon Semester 2027-28: ₹35,000
- Winter Semester 2027-28: ₹35,000

One Time Fees:
- Caution Deposit: ₹20,000
- Admission Fee: ₹5,000
- Campus Development Fee: ₹25,000
- Alumni Affairs Fee: ₹5,000
- Convocation Fee: ₹5,000
- Identity Card & Miscellaneous Fees: ₹1,000
- Total of One Time Fees: ₹61,000

Note:
Tuition fee is exempted for SC/ST students.
""",
        metadata={
            "source": "fee_structure.pdf",
            "category": "M.Tech/M.Plan",
            "programme": "M.Tech / M.Plan",
            "fee_type": "academic fees"
        }
    ),
    Document(
        page_content="""
NIT CALICUT — ACADEMIC FEES FOR 2026-27 ADMISSION BATCH

Programme: M.Sc.
Category: Students admitted through CCMN
Tuition Fee: ₹7,500 per semester (Monsoon and Winter semesters).

Programme: MBA (Regular - CAT)
Tuition Fee: ₹75,000 per semester.

Programme: Ph.D. (Full Time)
Tuition Fee: ₹7,500 per semester.

NOTE: This fee structure document does not contain undergraduate (B.Tech / B.Arch) fee structures.
""",
        metadata={
            "source": "fee_structure.pdf",
            "category": "Other PG Programmes",
            "fee_type": "academic fees"
        }
    )
]

structured_calendar_documents = [
    Document(
        page_content="""
NIT CALICUT — ACADEMIC CALENDAR FOR MONSOON SEMESTER 2026-27

Important dates:
1. Registration to Monsoon 2026 and Fee Payment without fine: 01.07.2026 to 17.07.2026
2. Monsoon 2026 Enrolment Day (Mandatory Physical Reporting): 20.07.2026
3. First Instructional Day: 21.07.2026
4. Late Registration: 30.07.2026
5. Last Date for Add/Drop Courses: 30.07.2026
6. Institute Foundation Day: 01.09.2026
7. Last Instructional Day: 10.11.2026
8. Result Declaration: 10.12.2026
""",
        metadata={
            "source": "academic_calendar.pdf",
            "semester": "Monsoon Semester 2026-27",
            "category": "academic calendar"
        }
    )
]

structured_admission_documents = [
    Document(
        page_content="""
NIT CALICUT — PHYSICAL REPORTING AND ADMISSION INFORMATION
Academic Year 2026-27

Applicable to candidates who got final seat allotment through CCMT/CCMN for M.Tech./M.Plan/M.Sc. programmes.

Physical Reporting Venue:
Bhaskara Hall, NIT Calicut.

Physical Reporting Schedule:
1. Department of Architecture & Planning and Department of Civil Engineering:
   Date: 05 August 2026
   Time: 9:30 am
2. Department of Computer Science & Engineering and Department of Electronics & Communication Engineering:
   Date: 06 August 2026
   Time: 9:30 am

Documents to be Produced in Original at the Time of Physical Reporting:
- Proof of Date of Birth: Class X marksheet/certificate issued by the school last attended or recognized educational board containing date of birth.
- Provisional Seat Allotment letter.
- Qualifying degree marksheets and degree/provisional certificate.
- Photo identity proof (Aadhaar, Passport, etc.).
- Category certificate (SC/ST/OBC-NCL/GEN-EWS/PwD), if applicable.
- Transfer certificate and Conduct/Character certificate from the institution last attended.
""",
        metadata={
            "source": "admission_rules.pdf",
            "category": "admission",
            "year": "2026-27"
        }
    )
]

structured_attendance_documents = [
    Document(
        page_content="""
NIT CALICUT — ATTENDANCE RULES

Attendance Requirement:
Students whose attendance is less than 80% or the limit prescribed by the course faculty for any course registered in a semester shall be informed of the shortage of attendance on or before the last instructional day.

Attendance Shortage and Condonation:
Students with attendance less than 80% or the limit prescribed by the course faculty for any course registered in a semester may be eligible to get their shortage of attendance condoned and hence appear for the end semester examination for that course only if they apply for condonation and satisfy the prescribed conditions.

One stated condition is:
The attendance in that semester for the course concerned, without applying any condonation, must not be less than 50% of the total classes handled for that course.

Attendance Calculation:
Attendance is counted from the date of commencement of the semester as per the academic calendar, except for first semester students.
For first-semester students, attendance is counted from the date of admission to the Institute or the start of classes, whichever is later.
""",
        metadata={
            "source": "student_handbook.pdf",
            "category": "attendance"
        }
    )
]


# --------------------------------------------------
# VECTOR STORE INITIALIZATION
# --------------------------------------------------

def load_pdf_chunks(filename: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    path = os.path.join(DOCUMENTS_PATH, filename)
    if not os.path.exists(path):
        logger.warning("Document file %s not found.", path)
        return []
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source"] = filename
        return chunks
    except Exception as e:
        logger.error("Failed to load PDF %s: %s", path, e)
        return []


def create_or_load_vectorstore(collection_name: str, documents: List[Document]) -> Chroma:
    """
    Creates or reloads a persistent Chroma vector store collection.
    """
    try:
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        # Check if already populated
        existing = vs.get()
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            return vs
    except Exception:
        pass

    # Create new collection from documents
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR
    )


# Prepare combined document lists
exam_all_docs = structured_exam_documents + load_pdf_chunks("exam_malpractice_guidelines.pdf")
fee_all_docs = structured_fee_documents + load_pdf_chunks("fee_structure.pdf")
calendar_all_docs = structured_calendar_documents + load_pdf_chunks("academic_calendar.pdf")
admission_all_docs = structured_admission_documents + load_pdf_chunks("admission_rules.pdf")
attendance_all_docs = structured_attendance_documents + load_pdf_chunks("student_handbook.pdf")

# Create Vector Stores
exam_vectorstore = create_or_load_vectorstore("college_helpdesk_final_exam", exam_all_docs)
fee_vectorstore = create_or_load_vectorstore("college_helpdesk_final_fees", fee_all_docs)
calendar_vectorstore = create_or_load_vectorstore("college_helpdesk_final_calendar", calendar_all_docs)
admission_vectorstore = create_or_load_vectorstore("college_helpdesk_final_admission", admission_all_docs)
attendance_vectorstore = create_or_load_vectorstore("college_helpdesk_final_attendance", attendance_all_docs)

# Retrievers
exam_retriever = exam_vectorstore.as_retriever(search_kwargs={"k": 3})
fee_retriever = fee_vectorstore.as_retriever(search_kwargs={"k": 4})
calendar_retriever = calendar_vectorstore.as_retriever(search_kwargs={"k": 3})
admission_retriever = admission_vectorstore.as_retriever(search_kwargs={"k": 3})
attendance_retriever = attendance_vectorstore.as_retriever(search_kwargs={"k": 3})


# --------------------------------------------------
# CONTEXT & HISTORY FORMATTING HELPERS
# --------------------------------------------------

def format_history(history: Optional[List[dict]], max_turns: int = 3) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        q = turn.get("question", "")
        a = turn.get("answer", "")
        if q and a:
            lines.append(f"Student: {q}\nHelpdesk: {a}")
    if not lines:
        return ""
    return "Recent Conversation History:\n" + "\n---\n".join(lines) + "\n\n"


def format_context_and_sources(docs: List[Document]) -> Tuple[str, List[str]]:
    if not docs:
        return "", []
    context_blocks = []
    sources = []
    for idx, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "college_document.pdf")
        if src not in sources:
            sources.append(src)
        context_blocks.append(f"[Document {idx} - Source: {src}]\n{doc.page_content.strip()}")
    return "\n\n".join(context_blocks), sources


# --------------------------------------------------
# LLM-POWERED ROUTER
# --------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are an intelligent query router for the National Institute of Technology Calicut (NITC) Student Helpdesk.
Classify the student's question into EXACTLY ONE of the following 5 categories:

1. exam: Examination rules, malpractice, punishments (e.g. impersonation, copying, debarment, F grade, REX/SAY/supplementary exams).
2. fee: Tuition fees, semester fees, one-time fees, caution deposit, fee exemptions, payment deadlines.
3. admission: Physical reporting venue, reporting dates, schedule by department, required documents/certificates for admission, CCMT/CCMN seat allotment.
4. calendar: Academic calendar dates, semester start/first instructional day, enrolment day, Add/Drop deadline, late registration, last instructional day, result declaration.
5. general: Attendance rules (80% requirement, shortage, condonation, attendance calculation), general student handbook policies, or queries not clearly matching the others.

Follow-up questions:
If the question is a follow-up (e.g. "What about the winter semester?"), use the conversation history to identify the topic (e.g. fee or calendar).

Output ONLY the category name in lowercase: exam, fee, admission, calendar, or general. Do not add any explanation or punctuation."""


def fallback_keyword_router(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["exam", "examination", "malpractice", "impersonation", "copying", "cheating", "rex", "say", "supplementary"]):
        return "exam"
    if any(w in q for w in ["fee", "fees", "tuition", "cost", "charges", "deposit", "payment", "amount"]):
        return "fee"
    if any(w in q for w in ["admission", "admissions", "reporting", "venue", "documents", "document", "certificate", "seat allotment", "physical reporting"]):
        return "admission"
    if any(w in q for w in ["semester", "instructional day", "calendar", "enrolment", "enrollment", "add/drop", "add drop", "registration date", "result declaration", "commence", "begin"]):
        return "calendar"
    return "general"


def router_node(state: dict) -> dict:
    question = state["question"]
    history_str = format_history(state.get("history", []))

    user_content = f"{history_str}Current Student Question: {question}"

    category = "general"
    try:
        response = invoke_llm_with_fallback([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ], temperature=0.0)

        cleaned = response.strip().lower()
        # Look for category in response
        for cat in ["exam", "fee", "admission", "calendar", "general"]:
            if cat in cleaned:
                category = cat
                break
    except Exception as e:
        logger.warning("LLM router failed (%s). Falling back to keyword router.", e)
        category = fallback_keyword_router(question)

    return {"category": category}


def route_question(state: dict) -> str:
    return state.get("category", "general")


# --------------------------------------------------
# SPECIALIZED AGENT GENERATION HELPER
# --------------------------------------------------

def generate_agent_response(
    agent_name: str,
    system_instructions: str,
    retriever: Any,
    state: dict
) -> dict:
    question = state["question"]
    history = state.get("history", [])

    # Retrieve relevant document chunks
    try:
        retrieved_docs = retriever.invoke(question)
    except Exception as e:
        logger.error("%s retriever failed: %s", agent_name, e)
        retrieved_docs = []

    context_str, sources = format_context_and_sources(retrieved_docs)
    history_str = format_history(history)

    user_prompt = f"""{history_str}Retrieved College Document Context:
{context_str if context_str else "No documents found."}

Student Question: {question}

Provide a concise, grounded, and factual answer based strictly on the retrieved college documents."""

    try:
        answer = invoke_llm_with_fallback([
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_prompt)
        ], temperature=0.0)
    except Exception as e:
        logger.error("%s LLM invocation failed: %s", agent_name, e)
        # Resilient fallback if API call fails
        if not retrieved_docs:
            answer = "I could not find information regarding this question in the available college documents."
        else:
            answer = (
                f"Information from {', '.join(sources)}:\n\n"
                f"{retrieved_docs[0].page_content.strip()}"
            )

    return {
        "answer": answer.strip(),
        "sources": sources
    }


# --------------------------------------------------
# SPECIALIZED AGENTS
# --------------------------------------------------

EXAM_SYSTEM_PROMPT = """You are the Examination and Malpractice Guidelines Agent for National Institute of Technology Calicut.
Your role is to give factual, accurate, and concise answers to student questions based strictly on the retrieved college examination guidelines.

Strict Grounding Rules:
1. Answer ONLY using information supported by the retrieved document context.
2. For examination malpractice cases (e.g. impersonation, copying, REX/SAY/supplementary exams), state the exact prescribed punishments from the guidelines.
3. If the user asks about an examination policy or situation not covered in the retrieved documents, clearly state that the information was not found in the examination guidelines.
4. Do not invent penalties, procedures, or grades."""


def exam_agent(state: dict) -> dict:
    return generate_agent_response("Exam Agent", EXAM_SYSTEM_PROMPT, exam_retriever, state)


FEE_SYSTEM_PROMPT = """You are the Academic Fee Structure Agent for National Institute of Technology Calicut.
Your role is to give factual, accurate, and concise answers to student fee questions based strictly on the retrieved fee documents.

CRITICAL Anti-Hallucination and Grounding Rules:
1. Answer ONLY using information directly supported by the retrieved college documents.
2. If the user asks about a programme (e.g., B.Tech, B.Arch), category, or semester that is NOT documented in the retrieved context, state clearly and unequivocally that fee information for that programme could not be found in the available college documents.
3. NEVER substitute, assume, or infer the fee of one programme (such as M.Tech or M.Plan) for another programme (such as B.Tech).
4. Preserve exact fee amounts, semester distinctions (e.g., Monsoon vs Winter), one-time fees, and exemption rules (e.g., tuition fee exemption for SC/ST students).
5. If the evidence is ambiguous or absent, clearly state the limitation without guessing."""


def fee_agent(state: dict) -> dict:
    return generate_agent_response("Fee Agent", FEE_SYSTEM_PROMPT, fee_retriever, state)


ADMISSION_SYSTEM_PROMPT = """You are the Admission and Physical Reporting Agent for National Institute of Technology Calicut.
Your role is to give factual, accurate, and concise answers regarding physical reporting, admission schedules, reporting venue, and required documents.

Strict Grounding Rules:
1. Answer ONLY using information supported by the retrieved admission documents.
2. Specify the exact physical reporting venue (Bhaskara Hall, NIT Calicut) and departmental reporting dates and times when asked.
3. List the exact original documents required (such as Class X marksheet for proof of date of birth, provisional seat allotment letter, qualifying degree certificates, etc.).
4. If the requested information is absent from the documents, state that it could not be found in the available admission documents."""


def admission_agent(state: dict) -> dict:
    return generate_agent_response("Admission Agent", ADMISSION_SYSTEM_PROMPT, admission_retriever, state)


CALENDAR_SYSTEM_PROMPT = """You are the Academic Calendar Agent for National Institute of Technology Calicut.
Your role is to give factual, accurate, and concise answers about semester dates, enrolment, Add/Drop deadlines, instructional days, and result declarations.

Strict Grounding Rules:
1. Answer ONLY using dates and details stated in the retrieved academic calendar.
2. State the exact dates (e.g. First Instructional Day: 21 July 2026, Enrolment Day: 20 July 2026, Add/Drop Deadline: 30 July 2026, Last Instructional Day: 10 November 2026, Result Declaration: 10 December 2026).
3. If an event or date is not listed in the calendar, state that it is not available in the academic calendar."""


def calendar_agent(state: dict) -> dict:
    return generate_agent_response("Calendar Agent", CALENDAR_SYSTEM_PROMPT, calendar_retriever, state)


GENERAL_SYSTEM_PROMPT = """You are the General Help and Attendance Policy Agent for National Institute of Technology Calicut.
Your role is to answer questions about attendance rules, condonation, attendance shortage, and student handbook policies based strictly on the retrieved documents.

Strict Grounding Rules:
1. Answer ONLY using information supported by the retrieved documents.
2. State the exact attendance rules:
   - Students with attendance below 80% (or the limit prescribed by faculty) are informed of attendance shortage on or before the last instructional day.
   - Attendance shortage condonation requires that attendance without condonation is NOT less than 50% of total classes handled.
   - Attendance calculation starts from the semester commencement date, or for first semester students, from the date of admission or start of classes, whichever is later.
3. If the answer is not supported by the retrieved context, state clearly that the information could not be found in the college documents."""


def general_agent(state: dict) -> dict:
    return generate_agent_response("General Agent", GENERAL_SYSTEM_PROMPT, attendance_retriever, state)


# --------------------------------------------------
# STATE & HISTORY
# --------------------------------------------------

class HelpdeskState(TypedDict):
    question: str
    category: str
    answer: str
    history: list
    sources: list


def update_history(state: dict) -> dict:
    history = state.get("history", []) or []
    history.append({
        "question": state["question"],
        "answer": state["answer"],
        "category": state.get("category", "general"),
        "sources": state.get("sources", [])
    })
    return {"history": history}


# --------------------------------------------------
# PYDANTIC RESPONSE VALIDATION
# --------------------------------------------------

class HelpdeskResponse(BaseModel):
    category: Literal[
        "exam",
        "fee",
        "admission",
        "calendar",
        "general"
    ]
    answer: str = Field(
        description="The final grounded answer given to the student."
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of document filenames consulted."
    )


# --------------------------------------------------
# LANGGRAPH WORKFLOW
# --------------------------------------------------

graph_builder = StateGraph(HelpdeskState)

graph_builder.add_node("router", router_node)
graph_builder.add_node("exam", exam_agent)
graph_builder.add_node("fee", fee_agent)
graph_builder.add_node("admission", admission_agent)
graph_builder.add_node("calendar", calendar_agent)
graph_builder.add_node("general", general_agent)
graph_builder.add_node("update_history", update_history)

# START -> router
graph_builder.add_edge(START, "router")

# Router -> Conditional Edge -> Specialized Agent
graph_builder.add_conditional_edges(
    "router",
    route_question,
    {
        "exam": "exam",
        "fee": "fee",
        "admission": "admission",
        "calendar": "calendar",
        "general": "general"
    }
)

# Agents -> update_history
graph_builder.add_edge("exam", "update_history")
graph_builder.add_edge("fee", "update_history")
graph_builder.add_edge("admission", "update_history")
graph_builder.add_edge("calendar", "update_history")
graph_builder.add_edge("general", "update_history")

# update_history -> END
graph_builder.add_edge("update_history", END)

# Compile Graph
graph = graph_builder.compile()


# --------------------------------------------------
# MAIN HELPDESK ENTRY POINT
# --------------------------------------------------

def run_helpdesk(question: str, history: Optional[list] = None) -> Tuple[HelpdeskResponse, list]:
    """
    Main entry point for College Helpdesk Multi-Agent RAG.
    Executes LangGraph workflow and returns validated Pydantic response with updated history.
    """
    if history is None:
        history = []

    result = graph.invoke({
        "question": question,
        "category": "",
        "answer": "",
        "history": history,
        "sources": []
    })

    # Validate response using Pydantic
    try:
        validated_response = HelpdeskResponse(
            category=result["category"],
            answer=result["answer"],
            sources=result.get("sources", [])
        )
    except Exception as e:
        logger.error("Pydantic validation error: %s. Using safe fallback.", e)
        validated_response = HelpdeskResponse(
            category="general",
            answer=str(result.get("answer", "I encountered an issue processing your request.")),
            sources=result.get("sources", [])
        )

    return validated_response, result["history"]
