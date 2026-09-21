
import streamlit as st
from backend import run_helpdesk


st.set_page_config(
    page_title="College Helpdesk",
    page_icon="🎓"
)


st.title("🎓 College Helpdesk")

st.write(
    "Ask questions about fees, exams, admissions, "
    "academic calendar, or attendance."
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
            st.caption(
                f"Agent: {message['category']}"
            )


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask your question..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # Display user question
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # Run multi-agent helpdesk
    response, history = run_helpdesk(
        question,
        st.session_state.history
    )

    st.session_state.history = history


    # Display assistant response
    with st.chat_message("assistant"):

        st.write(response.answer)

        st.caption(
            f"Agent: {response.category}"
        )


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "category": response.category
    })
