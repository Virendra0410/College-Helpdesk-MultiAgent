"""
Test suite for College Helpdesk Multi-Agent RAG System.
Includes:
- Unit tests with mock responses for fast CI execution
- Integration tests covering the 10 prompt-specified queries:
    1. Punishment for impersonation
    2. M.Tech tuition fee
    3. B.Tech fee (anti-hallucination check)
    4. Venue for physical reporting
    5. Semester begin date
    6. Attendance below 80%
    7. Last date for Add/Drop
    8. Admission documents
    9. Absent information
    10. Multi-turn follow-up question
"""

import pytest
from unittest.mock import patch, MagicMock
from backend import run_helpdesk, HelpdeskResponse, fallback_keyword_router


# =====================================================================
# 1. ROUTER UNIT TESTS (Fallback and Category Matching)
# =====================================================================

def test_fallback_router_exam():
    assert fallback_keyword_router("What is the penalty for cheating in an exam?") == "exam"
    assert fallback_keyword_router("Tell me about impersonation rules") == "exam"


def test_fallback_router_fee():
    assert fallback_keyword_router("What is the tuition fee for M.Tech?") == "fee"
    assert fallback_keyword_router("What is the caution deposit amount?") == "fee"


def test_fallback_router_admission():
    assert fallback_keyword_router("Where is the venue for physical reporting?") == "admission"
    assert fallback_keyword_router("What certificates and documents are needed?") == "admission"


def test_fallback_router_calendar():
    assert fallback_keyword_router("When is the last date for add/drop courses?") == "calendar"
    assert fallback_keyword_router("When is enrolment day?") == "calendar"


def test_fallback_router_general():
    assert fallback_keyword_router("What is the hostel policy?") == "general"
    assert fallback_keyword_router("What are the library hours?") == "general"


# =====================================================================
# 2. PYDANTIC VALIDATION TESTS
# =====================================================================

def test_pydantic_valid_response():
    resp = HelpdeskResponse(
        category="fee",
        answer="The M.Tech tuition fee is ₹35,000.",
        sources=["fee_structure.pdf"]
    )
    assert resp.category == "fee"
    assert "₹35,000" in resp.answer
    assert resp.sources == ["fee_structure.pdf"]


def test_pydantic_invalid_category():
    with pytest.raises(Exception):
        HelpdeskResponse(
            category="invalid_category",  # type: ignore
            answer="Some answer"
        )


# =====================================================================
# 3. ANTI-HALLUCINATION UNIT TEST (MOCK)
# =====================================================================

@patch("backend.invoke_llm_with_fallback")
def test_btech_fee_anti_hallucination_mock(mock_llm):
    # Mock LLM returns grounded rejection
    mock_llm.side_effect = [
        "fee",  # router output
        "Fee information for the B.Tech programme could not be found in the available college documents."
    ]

    response, _ = run_helpdesk("What is the fee for B.Tech?")
    assert response.category == "fee"
    assert "35,000" not in response.answer
    assert "could not be found" in response.answer.lower() or "not available" in response.answer.lower()


# =====================================================================
# 4. LIVE INTEGRATION TESTS (ALL 10 REQUIRED BENCHMARKS)
# =====================================================================

class TestCollegeHelpdeskLive:
    """
    Live integration tests verifying grounding against the college PDFs.
    Requires GOOGLE_API_KEY in environment or .env.
    """

    def test_q1_impersonation_punishment(self):
        """1. What is the punishment for impersonation in an examination?"""
        res, _ = run_helpdesk("What is the punishment for impersonation in an examination?")
        assert res.category == "exam"
        ans = res.answer.lower()
        # Item 11: 'F' grade in all subjects AND debarred for next two subsequent semesters
        assert "f" in ans
        assert "debarred" in ans or "two" in ans

    def test_q2_mtech_tuition_fee(self):
        """2. How much is the tuition fee for M.Tech?"""
        res, _ = run_helpdesk("How much is the tuition fee for M.Tech?")
        assert res.category == "fee"
        ans = res.answer.replace(",", "")
        assert "35000" in ans or "35,000" in res.answer

    def test_q3_btech_fee_anti_hallucination(self):
        """3. What is the fee for B.Tech? (MUST NOT return M.Tech fee of 35,000)"""
        res, _ = run_helpdesk("What is the fee for B.Tech?")
        assert res.category == "fee"
        ans = res.answer.lower()
        # Anti-hallucination assertion: must NEVER return 35,000 as B.Tech fee
        assert "35,000" not in res.answer and "35000" not in res.answer
        # Must clearly state that B.Tech fee information is absent/not found
        assert "not found" in ans or "not available" in ans or "could not be found" in ans

    def test_q4_physical_reporting_venue(self):
        """4. Where is the venue for physical reporting?"""
        res, _ = run_helpdesk("Where is the venue for physical reporting?")
        assert res.category == "admission"
        assert "bhaskara hall" in res.answer.lower()

    def test_q5_semester_begin_date(self):
        """5. When does the semester begin?"""
        res, _ = run_helpdesk("When does the semester begin?")
        assert res.category == "calendar"
        ans = res.answer.lower()
        assert "21" in ans and ("july" in ans or "07" in ans)

    def test_q6_attendance_below_80(self):
        """6. What happens if my attendance is below 80%?"""
        res, _ = run_helpdesk("What happens if my attendance is below 80%?")
        assert res.category == "general"
        ans = res.answer.lower()
        assert "shortage" in ans or "condonation" in ans or "50%" in ans or "last instructional day" in ans

    def test_q7_add_drop_last_date(self):
        """7. When is the last date for Add/Drop?"""
        res, _ = run_helpdesk("When is the last date for Add/Drop?")
        assert res.category == "calendar"
        ans = res.answer.lower()
        assert "30" in ans and ("july" in ans or "07" in ans)

    def test_q8_admission_documents(self):
        """8. What documents should I bring for admission?"""
        res, _ = run_helpdesk("What documents should I bring for admission?")
        assert res.category == "admission"
        ans = res.answer.lower()
        assert "marksheet" in ans or "birth" in ans or "original" in ans

    def test_q9_absent_information(self):
        """9. A question about information that is absent from all available documents."""
        res, _ = run_helpdesk("What are the timings for the campus swimming pool?")
        ans = res.answer.lower()
        assert "could not be found" in ans or "not found" in ans or "not available" in ans

    def test_q10_multi_turn_follow_up(self):
        """10. A follow-up question that depends on the previous message."""
        turn1_res, history = run_helpdesk("What is the tuition fee for M.Tech?")
        assert turn1_res.category == "fee"

        turn2_res, _ = run_helpdesk("What about the winter semester?", history=history)
        assert turn2_res.category == "fee"
        assert len(turn2_res.answer) > 10
