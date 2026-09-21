
from typing import TypedDict, Literal

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Document Path
# --------------------------------------------------

documents_path = "documents"


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load PDF Documents
# --------------------------------------------------

fee_loader = PyPDFLoader(
    f"{documents_path}/fee_structure.pdf"
)

calendar_loader = PyPDFLoader(
    f"{documents_path}/academic_calendar.pdf"
)

handbook_loader = PyPDFLoader(
    f"{documents_path}/student_handbook.pdf"
)

admission_loader = PyPDFLoader(
    f"{documents_path}/admission_rules.pdf"
)

exam_loader = PyPDFLoader(
    f"{documents_path}/exam_malpractice_guidelines.pdf"
)


fee_docs = fee_loader.load()
calendar_docs = calendar_loader.load()
handbook_docs = handbook_loader.load()
admission_docs = admission_loader.load()
exam_docs = exam_loader.load()


# --------------------------------------------------
# Text Splitting
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

fee_chunks = text_splitter.split_documents(fee_docs)
calendar_chunks = text_splitter.split_documents(calendar_docs)
handbook_chunks = text_splitter.split_documents(handbook_docs)
admission_chunks = text_splitter.split_documents(admission_docs)
exam_chunks = text_splitter.split_documents(exam_docs)


print("Documents loaded successfully.")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Document Path
# --------------------------------------------------

documents_path = "documents"


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load PDF Documents
# --------------------------------------------------

fee_loader = PyPDFLoader(
    f"{documents_path}/fee_structure.pdf"
)

calendar_loader = PyPDFLoader(
    f"{documents_path}/academic_calendar.pdf"
)

handbook_loader = PyPDFLoader(
    f"{documents_path}/student_handbook.pdf"
)

admission_loader = PyPDFLoader(
    f"{documents_path}/admission_rules.pdf"
)

exam_loader = PyPDFLoader(
    f"{documents_path}/exam_malpractice_guidelines.pdf"
)


fee_docs = fee_loader.load()
calendar_docs = calendar_loader.load()
handbook_docs = handbook_loader.load()
admission_docs = admission_loader.load()
exam_docs = exam_loader.load()


# --------------------------------------------------
# Text Splitting
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

fee_chunks = text_splitter.split_documents(fee_docs)
calendar_chunks = text_splitter.split_documents(calendar_docs)
handbook_chunks = text_splitter.split_documents(handbook_docs)
admission_chunks = text_splitter.split_documents(admission_docs)
exam_chunks = text_splitter.split_documents(exam_docs)


print("Documents loaded successfully.")

# --------------------------------------------------
# STRUCTURED EXAM DOCUMENTS
# --------------------------------------------------

structured_documents = [
    Document(
        page_content="""
ITEM 11 — IMPERSONATION IN THE EXAMINATION

Nature of malpractice:
Impersonation in the examination (for both the students).

Punishment:
The concerned students shall be awarded 'F' grade in all subjects
credited in the corresponding semester.

AND

The concerned students shall be debarred from attending classes
and appearing for examinations in the next two subsequent semesters.
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

In addition, the student(s) shall be awarded one grade less for all the
other theory courses registered in the corresponding REX/SAY/Supplementary
examinations.

AND

The student(s) shall be debarred from the provision of any
REX/SAY/Supplementary examinations in future.
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
Punishment will be as recommended by the Department level committee
for malpractice enquiry or Institute level committee for malpractice enquiry.
""",
        metadata={
            "source": "exam_malpractice_guidelines.pdf",
            "item": 13,
            "category": "examination malpractice"
        }
    )
]


# --------------------------------------------------
# STRUCTURED FEE DOCUMENT
# --------------------------------------------------

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
            "fee_type": "academic fees"
        }
    )
]


# --------------------------------------------------
# STRUCTURED CALENDAR DOCUMENT
# --------------------------------------------------

structured_calendar_documents = [
    Document(
        page_content="""
NIT CALICUT — ACADEMIC CALENDAR FOR MONSOON SEMESTER 2026-27

Important dates:

1. Registration to Monsoon 2026 and Fee Payment without fine:
   01.07.2026 to 17.07.2026

2. Monsoon 2026 Enrolment Day (Mandatory Physical Reporting):
   20.07.2026

3. First Instructional Day:
   21.07.2026

4. Late Registration:
   30.07.2026

5. Last Date for Add/Drop Courses:
   30.07.2026

6. Institute Foundation Day:
   01.09.2026

7. Last Instructional Day:
   10.11.2026

8. Result Declaration:
   10.12.2026
""",
        metadata={
            "source": "academic_calendar.pdf",
            "semester": "Monsoon Semester 2026-27",
            "category": "academic calendar"
        }
    )
]


print("Structured documents added.")

# --------------------------------------------------
# STRUCTURED ADMISSION DOCUMENT
# --------------------------------------------------

structured_admission_documents = [
    Document(
        page_content="""
NIT CALICUT — PHYSICAL REPORTING AND ADMISSION INFORMATION
Academic Year 2026-27

Applicable to candidates who got final seat allotment through CCMT/CCMN
for M.Tech./M.Plan/M.Sc. programmes.

Physical Reporting Venue:
Bhaskara Hall, NIT Calicut.

Physical Reporting Schedule:

1. Department of Architecture & Planning
   Department of Civil Engineering
   Date: 05 August 2026
   Time: 9:30 am

2. Department of Computer Science & Engineering
   Department of Electronics & Communication Engineering
   Date: 06 August 2026
   Time: 9:30 am

Documents to be Produced at the Time of Physical Reporting:

Candidates are required to bring the following documents in original.

Proof of Date of Birth:
Class X marksheet/certificate issued by the school last attended
or recognized educational board containing the date of birth of the applicant.

The admission document contains additional documents required during
physical reporting.
""",
        metadata={
            "source": "admission_rules.pdf",
            "category": "admission",
            "year": "2026-27"
        }
    )
]


# --------------------------------------------------
# STRUCTURED ATTENDANCE DOCUMENT
# --------------------------------------------------

structured_attendance_documents = [
    Document(
        page_content="""
NIT CALICUT — ATTENDANCE RULES

Attendance Requirement:

Students whose attendance is less than 80% or the limit prescribed
by the course faculty for any course registered in a semester shall
be informed of the shortage of attendance on or before the last
instructional day.

Attendance Shortage and Condonation:

Students with attendance less than 80% or the limit prescribed by
the course faculty for any course registered in a semester may be
eligible to get their shortage of attendance condoned and hence
appear for the end semester examination for that course only if
they apply for condonation and satisfy the prescribed conditions.

One stated condition is:

The attendance in that semester for the course concerned, without
applying any condonation, must not be less than 50% of the total
classes handled for that course.

Attendance Calculation:

Attendance is counted from the date of commencement of the semester
as per the academic calendar, except for first semester students.

For first-semester students, attendance is counted from the date of
admission to the Institute or the start of classes, whichever is later.
""",
        metadata={
            "source": "student_handbook.pdf",
            "category": "attendance"
        }
    )
]


print("Admission and attendance documents added.")

# --------------------------------------------------
# STRUCTURED VECTOR STORES
# --------------------------------------------------

exam_vectorstore = Chroma.from_documents(
    documents=structured_documents,
    embedding=embeddings,
    collection_name="college_helpdesk_final_exam"
)

exam_retriever = exam_vectorstore.as_retriever(
    search_kwargs={"k": 1}
)


fee_vectorstore = Chroma.from_documents(
    documents=structured_fee_documents,
    embedding=embeddings,
    collection_name="college_helpdesk_final_fees"
)

fee_retriever = fee_vectorstore.as_retriever(
    search_kwargs={"k": 1}
)


calendar_vectorstore = Chroma.from_documents(
    documents=structured_calendar_documents,
    embedding=embeddings,
    collection_name="college_helpdesk_final_calendar"
)

calendar_retriever = calendar_vectorstore.as_retriever(
    search_kwargs={"k": 1}
)


admission_vectorstore = Chroma.from_documents(
    documents=structured_admission_documents,
    embedding=embeddings,
    collection_name="college_helpdesk_final_admission"
)

admission_retriever = admission_vectorstore.as_retriever(
    search_kwargs={"k": 1}
)


attendance_vectorstore = Chroma.from_documents(
    documents=structured_attendance_documents,
    embedding=embeddings,
    collection_name="college_helpdesk_final_attendance"
)

attendance_retriever = attendance_vectorstore.as_retriever(
    search_kwargs={"k": 1}
)


print("All structured vector stores created.")

# --------------------------------------------------
# EXAM ANSWER
# --------------------------------------------------

def get_exam_answer(question):
    results = exam_retriever.invoke(question)

    if not results:
        return (
            "I could not find this information in the available "
            "examination guidelines."
        )

    question_lower = question.lower()

    # Item 11 — Impersonation
    if "impersonation" in question_lower:
        return (
            "For impersonation in the examination, the concerned "
            "students shall be awarded F grade in all subjects "
            "credited in the corresponding semester. They shall also "
            "be debarred from attending classes and appearing for "
            "examinations in the next two subsequent semesters."
        )

    # Item 12 — REX/SAY/Supplementary malpractice
    if any(word in question_lower for word in [
        "rex",
        "say",
        "supplementary"
    ]):
        return (
            "For malpractice during REX/SAY/Supplementary examinations, "
            "the student shall be awarded F grade in the corresponding "
            "course. The student shall also be awarded one grade less "
            "for all other theory courses registered in the corresponding "
            "REX/SAY/Supplementary examinations and shall be debarred "
            "from future REX/SAY/Supplementary examinations."
        )

    # Item 13 — Other cases
    if (
        "other case" in question_lower
        or "not mentioned" in question_lower
        or "not listed" in question_lower
    ):
        return (
            "For any other case not mentioned in the guidelines, the "
            "punishment will be as recommended by the Department-level "
            "committee for malpractice enquiry or Institute-level "
            "committee for malpractice enquiry."
        )

    return results[0].page_content.strip()


# --------------------------------------------------
# FEE ANSWER
# --------------------------------------------------

def get_fee_answer(question):
    results = fee_retriever.invoke(question)

    if not results:
        return (
            "I could not find this information in the available "
            "college fee documents."
        )

    question_lower = question.lower()

    if "tuition fee" in question_lower:
        return (
            "The tuition fee for M.Tech/M.Plan is "
            "₹35,000 per listed semester."
        )

    return results[0].page_content.strip()


print("Exam and fee functions added.")

# --------------------------------------------------
# CALENDAR ANSWER
# --------------------------------------------------

def get_calendar_answer(question):
    results = calendar_retriever.invoke(question)

    if not results:
        return (
            "I could not find this information in the available "
            "academic calendar."
        )

    question_lower = question.lower()

    if "semester" in question_lower and (
        "begin" in question_lower
        or "start" in question_lower
        or "commence" in question_lower
    ):
        return "The first instructional day is 21 July 2026."

    if "last instructional day" in question_lower:
        return "The last instructional day is 10 November 2026."

    if "enrolment" in question_lower or "enrollment" in question_lower:
        return (
            "The Monsoon 2026 Enrolment Day "
            "(Mandatory Physical Reporting) is 20 July 2026."
        )

    if "add/drop" in question_lower:
        return "The last date for Add/Drop Courses is 30 July 2026."

    if "late registration" in question_lower:
        return "Late Registration is on 30 July 2026."

    if "result" in question_lower:
        return "The Result Declaration date is 10 December 2026."

    return results[0].page_content.strip()


# --------------------------------------------------
# ADMISSION ANSWER
# --------------------------------------------------

def get_admission_answer(question):
    results = admission_retriever.invoke(question)

    if not results:
        return (
            "I could not find this information in the available "
            "admission documents."
        )

    question_lower = question.lower()

    if "venue" in question_lower or "where" in question_lower:
        return (
            "The physical reporting venue is "
            "Bhaskara Hall, NIT Calicut."
        )

    if (
        "computer science" in question_lower
        or "electronics" in question_lower
    ):
        return (
            "The Department of Computer Science & Engineering and "
            "Department of Electronics & Communication Engineering "
            "have physical reporting on 06 August 2026 at 9:30 am."
        )

    if (
        "architecture" in question_lower
        or "civil" in question_lower
    ):
        return (
            "The Department of Architecture & Planning and "
            "Department of Civil Engineering have physical reporting "
            "on 05 August 2026 at 9:30 am."
        )

    if "document" in question_lower:
        return (
            "Candidates are required to bring the required documents "
            "in original during physical reporting. The admission "
            "document specifies proof of date of birth as a required "
            "document."
        )

    return results[0].page_content.strip()


print("Calendar and admission functions added.")

# --------------------------------------------------
# GENERAL / ATTENDANCE ANSWER
# --------------------------------------------------

def get_general_answer(question):
    results = attendance_retriever.invoke(question)

    if not results:
        return (
            "I could not find this information in the available "
            "college documents."
        )

    question_lower = question.lower()

    if (
        "when" in question_lower
        and (
            "shortage" in question_lower
            or "informed" in question_lower
        )
    ):
        return (
            "Students with attendance less than 80% or the limit "
            "prescribed by the course faculty shall be informed of "
            "the attendance shortage on or before the last "
            "instructional day."
        )

    if (
        "attendance requirement" in question_lower
        or "minimum attendance" in question_lower
        or "attendance required" in question_lower
    ):
        return (
            "Students whose attendance is less than 80% or the limit "
            "prescribed by the course faculty for a course shall be "
            "informed of the shortage of attendance on or before the "
            "last instructional day."
        )

    if (
        "below 80" in question_lower
        or "less than 80" in question_lower
        or "condonation" in question_lower
    ):
        return (
            "Students with attendance less than 80% or the limit "
            "prescribed by the course faculty may be eligible to get "
            "their shortage of attendance condoned and appear for "
            "the end semester examination for that course, provided "
            "they apply for condonation and satisfy the prescribed "
            "conditions. One stated condition is that attendance "
            "without condonation must not be less than 50% of the "
            "total classes handled for that course."
        )

    if (
        "attendance calculated" in question_lower
        or "attendance counted" in question_lower
        or "first semester" in question_lower
    ):
        return (
            "Attendance is counted from the date of commencement "
            "of the semester as per the academic calendar, except "
            "for first-semester students. For first-semester "
            "students, attendance is counted from the date of "
            "admission to the Institute or the start of classes, "
            "whichever is later."
        )

    return results[0].page_content.strip()


print("General/attendance function added.")

# --------------------------------------------------
# ROUTER
# --------------------------------------------------

def router_node(state):
    question = state["question"].lower()

    if any(word in question for word in [
        "exam",
        "examination",
        "malpractice",
        "impersonation",
        "copying"
    ]):
        category = "exam"

    elif any(word in question for word in [
        "fee",
        "fees",
        "payment",
        "tuition",
        "cost",
        "charges"
    ]):
        category = "fee"

    elif any(word in question for word in [
        "admission",
        "admissions",
        "apply",
        "application",
        "reporting",
        "documents",
        "document",
        "venue",
        "physical reporting",
        "seat allotment"
    ]):
        category = "admission"

    elif any(word in question for word in [
        "semester",
        "instructional day",
        "academic calendar",
        "enrolment",
        "enrollment",
        "add/drop",
        "registration date",
        "last instructional day",
        "first instructional day"
    ]):
        category = "calendar"

    else:
        category = "general"

    return {"category": category}


def route_question(state):
    return state["category"]


# --------------------------------------------------
# SPECIALIZED AGENTS
# --------------------------------------------------

def exam_agent(state):
    return {
        "answer": get_exam_answer(state["question"])
    }


def fee_agent(state):
    return {
        "answer": get_fee_answer(state["question"])
    }


def admission_agent(state):
    return {
        "answer": get_admission_answer(state["question"])
    }


def calendar_agent(state):
    return {
        "answer": get_calendar_answer(state["question"])
    }


def general_agent(state):
    return {
        "answer": get_general_answer(state["question"])
    }


print("Router and agents added.")

# --------------------------------------------------
# STATE
# --------------------------------------------------

class HelpdeskState(TypedDict):
    question: str
    category: str
    answer: str
    history: list


# --------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------

def update_history(state):
    history = state.get("history", [])

    history.append({
        "question": state["question"],
        "answer": state["answer"]
    })

    return {
        "history": history
    }


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
        description="The final answer given to the student."
    )


print("State, memory, and Pydantic validation added.")

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


# Start → Router
graph_builder.add_edge(
    START,
    "router"
)


# Router → Specialized Agent
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


# Specialized Agents → History
graph_builder.add_edge(
    "exam",
    "update_history"
)

graph_builder.add_edge(
    "fee",
    "update_history"
)

graph_builder.add_edge(
    "admission",
    "update_history"
)

graph_builder.add_edge(
    "calendar",
    "update_history"
)

graph_builder.add_edge(
    "general",
    "update_history"
)


# History → End
graph_builder.add_edge(
    "update_history",
    END
)


# Compile Graph
graph = graph_builder.compile()

print("LangGraph workflow created successfully.")

# --------------------------------------------------
# MAIN HELPDESK FUNCTION
# --------------------------------------------------

def run_helpdesk(question, history=None):

    if history is None:
        history = []

    result = graph.invoke({
        "question": question,
        "category": "",
        "answer": "",
        "history": history
    })

    # Validate final response using Pydantic
    validated_response = HelpdeskResponse(
        category=result["category"],
        answer=result["answer"]
    )

    return validated_response, result["history"]


print("run_helpdesk function added.")
