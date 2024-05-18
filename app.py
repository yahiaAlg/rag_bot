import streamlit as st
import os
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import FileChatMessageHistory, ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv
import asyncio


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return asyncio.get_event_loop()


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    st.set_page_config(
        page_title="Question Answer Bot",
        page_icon="🤖",
    )
    st.title("QA Bot")
    st.markdown("Please enter your query below:")
    question = st.text_area("🧑 query:", "")
    if st.button("Answer Question"):
        if not st.session_state.get("chain_bot", None):
            # Load the model
            # Then, before you instantiate ChatGoogleGenerativeAI, call this function:
            get_or_create_eventloop()
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)  # type: ignore
            # Create a prompt template
            prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(
                    input_variables=["question"],
                    template="Question: {question}\nAnswer: ",
                ),
            ])

            # Create a chain
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                chat_memory=FileChatMessageHistory("chat_history.json")
            )
            chain = LLMChain(llm=llm, prompt=prompt, memory=memory,verbose=True)
            st.session_state.chain_bot = chain
        # Run the Chain
        else:
            chain = st.session_state["chain_bot"]
        answer = chain.run(question)  # type: ignore
        # Display the result
        st.write("🤖 Answer:")
        st.write(f"{answer}")
    with st.sidebar:
        st.markdown(
            "This app was created by [Dr. John Doe](https://www.example.com).🐺",
            unsafe_allow_html=True,
        )  # type: ignore
        if "GOOGLE_API_KEY" not in os.environ:
            api_key = st.text_input("GOOGLE_API_KEY", type="password")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                st.success("Google APi key loaded!")
