# Personal Learning Assistant

A personal learning assistant built with langchain, and uses streamlit to display a 
simple UI. 

The chatbot can answer questions generally, and the app also has a provision to upload
notes and ask the chatbot questions about the uploaded notes. The main aim of the chatbot 
is to help students prepare for evaluations.

It is built with technologies including LangChain, GroqAI, Streamlit, and LangSmith.


## Features

- **Semantic Routing**: Dynamically routes questions based on the type (retrieval-based or general).
- **Chat History**: Maintains conversation history to provide contextually relevant responses.
- **Document Upload and Query**: Users can upload documents and ask questions directly related to the content.
- **LangSmith Integration**: Utilized for tracing and debugging to streamline the development process.
- **User Interface**: A simple and intuitive interface built with Streamlit.
- **LLM Integration**: Powered by the Llama3 LLM from the GroqAI API for robust language understanding.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DejusDevspace/Personal-Learning-Assistant-Q-A-Conversational-Chatbot.git
   ```
2. Navigate to the project directory:

    ```bash
   cd Personal-Learning-Assistant-Q-A-Conversational-Chatbot
    ```
3. Install the required dependencies:

    ```bash
   pip install -r requirements.txt
    ```

## Usage
To run the application locally:
```bash
streamlit run main.py
```
This will start the Streamlit server, and you can interact with the chatbot through the web interface.

## Technologies Used
- LangChain: For creating and managing the flow of LLMs.
- GroqAI: Provides the Llama3 LLM API for enhanced language processing.
- Streamlit: Simplifies the creation of the user interface.
- LangSmith: Used for tracing and debugging during the development process.

## Contact
For any questions or feedback, feel fre to reach out on here on GitHub or [LinkedIn](https://www.linkedin.com/in/deju-adejo)

## Screenshots
<img src="assets/images/Screenshot (129).png">
<hr>
<img src="assets/images/Screenshot (127).png">
<hr>
<img src="assets/images/Screenshot (131).png">

You can view more screenshots in the "assets/images/" directory
