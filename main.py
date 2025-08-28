import streamlit as st
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from questions import QUESTIONS
from config import NUMBER_OF_QUESTIONS
from utils import get_llm_reply, extract_data_from_session_state, stream_llm_response

#****************************** Set-up LLM and Utility Function ********************************
load_dotenv()
chat_model = ChatAnthropic(model="claude-3-5-haiku-20241022")

#****************************** PAGE CONFIG ***********************************************
st.set_page_config(page_title="TalentScout Interview Chatbot", page_icon="👋", layout="centered")

#****************************** CSS and LLM Prompt Template File***********************************************
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load system prompt from file
with open("InterviewerLLMPrompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

#****************************** SESSION STATE ***********************************************
# For details gathering section
if "stage" not in st.session_state:
    st.session_state.stage = "welcome"   # stages: welcome → details → interview → thankyou
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "question_index" not in st.session_state:
    st.session_state.question_index = 0

# For interview Section
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
if "step" not in st.session_state:
    st.session_state.step = 0
if "active" not in st.session_state:
    st.session_state.active = False
if "done" not in st.session_state:
    st.session_state.done = False    


#****************************** WELCOME SCREEN ***********************************************
if st.session_state.stage == "welcome":
    st.markdown('<p class="big-font">👋 Welcome </p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">I am an intelligent Hiring Assistant chatbot for <b>TalentScout</b>, a recruitment agency specializing in technology placements. </p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Here, your interview will be conducted. You need to provide us with a few details and we can get started with it</p>', unsafe_allow_html=True)

    if st.button(" Get Started"):
        st.session_state.stage = "details"
        st.rerun()


#****************************** DETAILS QUESTIONNAIRE ***********************************************
elif st.session_state.stage == "details":
    st.markdown('<header class="big-font"><h2>Kindly provide us with your details </h2></header>', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.message_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if st.session_state.question_index < len(QUESTIONS):

        if user_input := st.chat_input("Type your answer here..."):
            st.session_state.message_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            st.session_state.question_index += 1

        if st.session_state.question_index < len(QUESTIONS):
            next_question = QUESTIONS[st.session_state.question_index]

            # Stream reply
            response_text = get_llm_reply(st.session_state.message_history, next_question)
            st.session_state.message_history.append({"role": "assistant", "content": response_text})

        if st.session_state.question_index == len(QUESTIONS):
            final_message = "Thank you for answering all the questions! We will now move forward with the interview"
            st.chat_message("assistant").write(final_message)
            st.session_state.message_history.append({"role": "assistant", "content": final_message})

    if st.session_state.question_index >= len(QUESTIONS):
        if st.button("➡️ Go to Interview"):
            st.session_state.stage = "interview"
            st.rerun()


#****************************** INTERVIEW PAGE ***********************************************
elif st.session_state.stage == "interview":
    st.markdown('<p class="big-font">📝 Interview Stage</p>', unsafe_allow_html=True)
    st.write("Enter START to begin. If you want to quit the interview at any moment, enter EXIT")

    peronsal_details = extract_data_from_session_state(st.session_state.message_history)

    # Chat display
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # User Input
    if not st.session_state.done:
        user_input = st.chat_input("Type START to begin, or EXIT to quit...")

        if user_input:
            # Show user's message immediately
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            # Handle EXIT
            if user_input.strip().upper() == "EXIT":
                with st.chat_message("assistant"):
                    st.markdown("Thank you! Terminating the interview.")
                st.session_state.done = True

            # Handle START
            elif user_input.strip().upper() == "START" and not st.session_state.active:
                st.session_state.active = True
                st.session_state.step = 1

                # Generate first question using get_llm_reply
                with st.spinner("Generating first question..."):
                    response_text = get_llm_reply(
                        message_history=[
                            {"role": "user", "content": f"Tech stack: {peronsal_details.techstack}, "
                                                        f"Years Of Experience: {peronsal_details.experience}, "
                                                        f"Desired Position: {peronsal_details.positions}. "
                                                        "Start the Q&A."}
                        ],
                        next_question="Ask the first interview question."
                    )

                response = AIMessage(content=response_text)
                st.session_state.messages.append(response)

            # Handle Q&A flow
            elif st.session_state.active and st.session_state.step <= NUMBER_OF_QUESTIONS:
                #st.session_state.messages.append(HumanMessage(content=user_input))

                if st.session_state.step == NUMBER_OF_QUESTIONS:
                    # Last question answered → Thank you
                    with st.chat_message("assistant"):
                        st.markdown("✅ Thank you for your responses. Interview complete!")
                    st.session_state.done = True
                else:
                    # Continue with next question using get_llm_reply
                    with st.spinner("Generating next question..."):
                        response_text = get_llm_reply(
                            message_history=[
                                {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage)
                                else {"role": "assistant", "content": msg.content}
                                for msg in st.session_state.messages
                            ],
                            next_question=f"Ask question {st.session_state.step+1} of {NUMBER_OF_QUESTIONS}."
                        )

                    response = AIMessage(content=response_text)
                    st.session_state.messages.append(response)
                    st.session_state.step += 1

    if st.session_state.done == True:
        if st.button("I am Done"):
            st.session_state.stage = "thankyou"
            st.rerun()


#****************************** THANK YOU PAGE ***********************************************
elif st.session_state.stage == "thankyou":
    # Page title
    st.title("Thank You!")

    # Thank you message
    st.markdown('<p class="big-font">🎉 Your submission has been received successfully. </p>', unsafe_allow_html=True)

    # Add a small note or next steps
    st.write("You can now close this page.")