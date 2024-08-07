import streamlit as st
# from assistant import PersonalLearningAssistant
from langchain_core.messages import HumanMessage, AIMessage
from utils import tools, prompts
from learning_assistant import PersonalLearningAssistant

AIMessage(content=prompts.WELCOME_PROMPT)


def main():
    assistant = PersonalLearningAssistant()

    # --------------- PAGE CONFIGURATION --------------- #
    st.set_page_config(
        page_title="Assistant",
        page_icon="🤖",
    )

    st.title("Personal Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=prompts.WELCOME_PROMPT),
        ]

    uploaded_file = st.file_uploader("upload file", type=["pdf", "docx"], accept_multiple_files=False)
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_type = uploaded_file.name.split(".")[-1]

        # Upload the file to temporary location
        file_path = tools.load_file_to_dir(uploaded_file, suffix=file_type)
        # print(file_path, "-----", file_type)
        # Process the file depending on the file type
        documents = tools.process_file(file_path, file_type)
        # print(documents)
        # print("\n{}".format(len(documents)))
        retriever = assistant.load_vectorstore_as_retriever(documents)

        # test = retriever.invoke("screen sizes categories used in responsive design")
        # print(test, "\n", len(test))
        chain = assistant.initialize_retrieval_chain(retriever)
        response = chain.invoke({"chat_history": st.session_state.chat_history, "input": "Who is Jack Ma?"})

        print(response["answer"])


#     # ---------- SESSION STATE VARIABLES ---------- #
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#         st.session_state.chat_history = []
#
#         # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
#
#     # --------------- FILE HANDLING AND CHAT --------------- #
#     if uploaded_file is not None:
#         file_name = uploaded_file.name
#         file_type = uploaded_file.name.split(".")[-1]
#
#         st.write("File name:", file_name)
#         st.write("File type:", file_type)
#         st.divider()
#
#         if "documents" not in st.session_state:
#             st.session_state.documents = []
#
#         # Check if the uploaded file is new
#         # if file_name not in st.session_state.documents:
#         #     st.session_state.documents.append(file_name)
#         #     # Clear the cache
#         #     st.cache_resource.clear()
#         # if "messages" in st.session_state:
#         #     st.session_state.messages.clear()
#         #     st.session_state.chat_history.clear()
#
#         # print(file_name, "---", file_type)
#
#         # Initialize the vectorstore with the uploaded document
#         vectorstore = assistant.initialize_vectorstore(uploaded_file, file_type)
#         # retriever = assistant.initialize_retriever(vectorstore)
#
#         # test = assistant.vectorstore.similarity_search("How many useful lessons are in the text?")
#         # print(test)
#         # print("\n", len(test))
#         # if assistant.vectorstore is not None:
#         #     print("\n---Successfully loaded file!")
#         # Stream placeholder message only on reload...
#         # if "intro" not in st.session_state:
#         #     with st.chat_message('assistant'):
#         #         st.write_stream(tools.stream_data)
#         #     st.session_state.intro = prompts.WELCOME_PROMPT
#         #     st.session_state.messages.append({'role': 'assistant', 'content': st.session_state.intro})
#         # else:
#         #     with st.chat_message("assistant"):
#         #         st.markdown(st.session_state.intro)
#
#         retrieval_chain = assistant.initialize_retrieval_chain(vectorstore)
#
#         if prompt := st.chat_input("Ask me about your document..."):
#             # Display the user's query
#             with st.chat_message("user"):
#                 st.markdown(prompt)
#             # Display loading sequence while processing response
#             with st.spinner("Loading response..."):
#                 response = assistant.query(prompt, st.session_state.chat_history, chain=retrieval_chain)
#                 print(response)
#                 answer = response["answer"]
#
#                 # Display the assistant's message (stream)
#                 with st.chat_message("assistant"):
#                     stream_text = tools.stream_data(answer)
#                     st.write_stream(stream_text)
#                 # Append the message and response to the chat history
#                 st.session_state.messages.append({'role': 'user', 'content': prompt})
#                 st.session_state.messages.append({'role': 'assistant', 'content': answer})
#                 st.session_state.chat_history.extend([(prompt, answer)])


if __name__ == "__main__":
    main()
