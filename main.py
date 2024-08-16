import streamlit as st
from learning_assistant import PersonalLearningAssistant
from utils import tools, prompts


def load_css(path: str) -> None:
    """
    Loads a custom css file at for styling
    :param path: Path to the css file
    :return: None
    """
    with open(path, "r") as file:
        css = file.read()

    st.markdown(
        f"<style>{css}</style>", unsafe_allow_html=True)
    return None


def main():
    """Program entry function"""
    # ------------------------------ PAGE CONFIGURATION ------------------------------ #
    st.set_page_config(
        page_title="Assistant",
        page_icon="📝",
    )

    # Load styling from css file
    load_css("assets/styles/main.css")

    # ------------------------------ SIDEBAR ------------------------------ #
    # st.title("Learning assistant 📝 ")
    with st.sidebar:
        st.title("Settings")
        st.divider()
        st.markdown("""
                    This is the first version of my learning assistant. Current abilities include:

                    - Text Generation
                    - Text Summarization
                    - File Upload
                    - Learning Assistance

                    More functionalities to be added...
                    """)
        st.divider()

        # File upload
        uploaded_file = st.file_uploader("Upload your slides 📝", type=["pdf", "docx"])
        st.info(
            "On first query about uploaded file, make reference to the file in your prompt.\n\n"
            ":exclamation: Refresh to upload a new file!",
            icon=":material/info:")
        st.divider()
        st.markdown(
            """
            Visit my GitHub repository to view source code.

            Click the button below 👇
            """
        )
        st.link_button("DejusDevspace", "https://github.com/DejusDevspace/Personal-Leaning-Assistant-with-RAG-app")

    # ----------------------------- PAGE CONTENT ------------------------------ #
    st.markdown(
        """
        <div class="flex-container">
        <div class="container">

        </div>
        <div class="container">

        </div>
        <div class="container">

        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Made with :heart: by Deju")

    # SESSION STATE VARIABLES #
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": prompts.WELCOME_TEXT}]

    if "files" not in st.session_state:
        st.session_state.files = []

    if "assistant" not in st.session_state:
        st.session_state.assistant = PersonalLearningAssistant(model="llama3-70b-8192", temperature=0.4)

    # CONVERSATION CHAT HISTORY #
    # Display conversation history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # FILE UPLOAD
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
                st.error(
                    """
                    Unable to provide response at this time.

                    Possible fixes:

                        • Check your internet connection.

                        • Reload the page and try again.

                        • If you uploaded a file, make sure it is in an accepted format.

                    If the problem persists, please exit the page and try again later.
                    """
                )

        try:
            # Add new conversations to messages and chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Update the chat history
            st.session_state.chat_history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ])
        except Exception as e:
            print("Error:", e)

    # Removing some conversations to preserve context window
    if len(st.session_state.chat_history) > 5:
        # Get rid of the first message in the history when the messages are more than 5
        st.session_state.chat_history.pop(0)


if __name__ == "__main__":
    main()
