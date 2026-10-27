import streamlit as st
from backend import run_helpdesk
from llm import get_api_key, get_model_name


st.set_page_config(
    page_title="College Helpdesk — Multi-Agent RAG",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 College Helpdesk")
st.write(
    "Ask questions about examinations, fees, admissions, "
    "academic calendar, or attendance rules."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("Helpdesk Agents")
    st.markdown("""
    - **📝 Exam Agent:** Malpractice, grading penalties, impersonation, debarment.
    - **💰 Fee Agent:** Tuition fees, one-time fees, exemptions by programme.
    - **🏛️ Admission Agent:** Physical reporting venue, dates, required documents.
    - **📅 Calendar Agent:** Semester start/end, Add/Drop, enrolment dates.
    - **📖 General Agent:** Attendance requirements, shortage, condonation rules.
    """)

    st.divider()
    api_key_set = get_api_key() is not None
    if api_key_set:
        st.success(f"Connected to Gemini ({get_model_name()})")
    else:
        st.error("Gemini API key not configured")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------
# API KEY WARNING
# --------------------------------------------------
if not api_key_set:
    st.warning(
        "⚠️ **Google Gemini API Key Not Configured**\n\n"
        "To enable AI responses, add `GOOGLE_API_KEY` to your local `.env` file "
        "or configure it in Streamlit Community Cloud Secrets."
    )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            cat = message.get("category", "General").capitalize()
            sources = message.get("sources", [])
            src_str = f" | Sources: {', '.join(sources)}" if sources else ""
            st.caption(f"Agent: {cat}{src_str}")


# --------------------------------------------------
# USER INPUT & PROCESSING
# --------------------------------------------------

question = st.chat_input("Ask your question...")

if question:
    # Display user question immediately
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Run multi-agent helpdesk with spinner
    with st.chat_message("assistant"):
        with st.spinner("Consulting college documents and generating response..."):
            try:
                response, updated_history = run_helpdesk(
                    question,
                    st.session_state.history
                )
                st.session_state.history = updated_history

                st.write(response.answer)

                cat = response.category.capitalize()
                src_str = f" | Sources: {', '.join(response.sources)}" if response.sources else ""
                st.caption(f"Agent: {cat}{src_str}")

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "category": response.category,
                    "sources": response.sources
                })
            except Exception as e:
                err_msg = "An error occurred while processing your request. Please try again."
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "category": "general",
                    "sources": []
                })
