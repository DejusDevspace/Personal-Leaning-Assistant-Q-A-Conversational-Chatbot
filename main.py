import streamlit as st
from learning_assistant import PersonalLearningAssistant
from utils import tools


# assistant.route()
st.title("Learning assistant")
uploaded_file = st.file_uploader("upload file", type=["pdf", "docx"])

# -------------------- SESSION STATE VARIABLES -------------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = []

if "assistant" not in st.session_state:
    st.session_state.assistant = PersonalLearningAssistant(model="llama3-70b-8192", temperature=0.4)

# Display conversation history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file:
    file_name = uploaded_file.name
    file_type = uploaded_file.name.split(".")[-1]
    # Load a new vectorstore for new document uploads
    if file_name not in st.session_state.files:
        file_path = tools.load_file_to_dir(uploaded_file, suffix=file_type)
        documents = tools.process_file(file_path, file_type)
        st.session_state.assistant.load_vectorstore_as_retriever(documents)
        st.session_state.files.append(file_name)

if prompt := st.chat_input("Enter a prompt..."):
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display a loading sequence while generating response
    with st.spinner("Loading response..."):
        try:
            response = st.session_state.assistant.query(question=prompt, chat_history=st.session_state.chat_history)
            # Display the AI's message
            with st.chat_message("assistant"):
                stream = tools.stream_data(response)
                st.write_stream(stream)
        except Exception as e:
            print("Unable to load response:", e)
            st.error("Unable to provide response at this time.")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.extend([
        {"type": "human", "content": prompt},
        {"type": "assistant", "content": response}
    ])

# Removing some conversations to preserve context window size
if len(st.session_state.chat_history) > 8:
    # Get rid of the first message in the history when the messages get up to 8
    st.session_state.chat_history.pop(0)
