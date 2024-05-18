import os
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from typing import List
from django.conf import settings
# from google.colab import userdata


def get_api_keys():
    try:
        load_dotenv(find_dotenv(), override=True)
        return "successfully loaded the api keys"
    except Exception as e:
        return None


def load_docs_locally(files: List[str] = []):
    from pprint import pprint
    import os

    os.chdir(os.path.join(settings.BASE_DIR, "files/"))
    print(f"current directory: {os.getcwd()}")
    files = [file for file in os.listdir()] if not files else files
    pprint(files)

    data = []

    for file in files:
        _, extension = os.path.splitext(file)
        if not file.startswith("."):
            match extension:
                case ".pdf":
                    from langchain.document_loaders import PyPDFLoader

                    loader = PyPDFLoader(file)
                    print(f"loading pdf {file} ....")
                case ".txt":
                    from langchain.document_loaders import TextLoader

                    loader = TextLoader(file, encoding="utf-8")
                    print(f"loading text {file} ....")
                case ".docx":
                    from langchain.document_loaders import Docx2textLoader

                    loader = Docx2textLoader(file)
                    print(f"loading docx {file} ....")
                case _:
                    print(f"no such available format such as {extension}")

        data += loader.load()
    os.chdir(settings.BASE_DIR)
    return data


def download_file(url: str, filename: str):
    import requests, os

    binary_file = requests.get(url).content
    _, extension = os.path.splitext(url)

    with open(f"files/{filename}{extension}", "wb") as f:
        f.write(binary_file)

    print(f"done downloading {filename}{extension}")
    return f"files/{filename}{extension}"


def load_docs(docs_urls=["https://pypi.org/"]):
    from langchain.document_loaders.async_html import AsyncHtmlLoader

    print("loading started....")
    loader = AsyncHtmlLoader(docs_urls)
    documents = loader.load()
    return documents


def clean_html(html_page: str, title: str):
    from pprint import pprint
    from bs4 import BeautifulSoup

    parser = BeautifulSoup(html_page, "html.parser")
    # pprint(parser.prettify())
    with open(f"files/{title}.txt", "w", encoding="utf-8") as f:
        for string in parser.strings:
            if string != "\n":
                f.write(string.strip())
                f.write("\n")


def mass_download(urls: List[str]):
    file_titles = []
    html_pages = load_docs(urls)
    for i, html_page in enumerate(html_pages):
        cleaned_file_title = (
            urls[i]
            .replace("/", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("https:", "")
            .replace("dz", "")
            .replace("net", "")
            .replace("com", "")
            .replace("org", "")
            .replace("edu", "")
            .strip("_")
        )
        clean_html(html_page.page_content, cleaned_file_title)
        file_titles.append(cleaned_file_title)
    return file_titles


def chunk_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = "\n".join([doc.page_content for doc in docs])
    # print(text)
    chunks = text_splitter.split_text(text)
    return chunks


def insert_or_create_index(index_name, chunks):
    import pinecone
    from pinecone import PodSpec
    from langchain_community.vectorstores.pinecone import Pinecone
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

    # os.environ["PINECONE_API_KEY"] = userdata.get("PINECONE_API_KEY")
    pc = pinecone.Pinecone()

    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  # type: ignore
    if index_name in pc.list_indexes().names():
        print(f"start fetching from {index_name}!")
        vector_store = Pinecone.from_existing_index(index_name, embedding)
        print(f"done fetching from {index_name}!")
    else:
        print(f"start creating from {index_name}!")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )
        vector_store = Pinecone.from_texts(chunks, embedding, index_name=index_name)
        print(f"done creation of {index_name}!")
    return vector_store


def delete_index(index_name="all"):
    from pinecone import Pinecone

    # os.environ["PINECONE_API_KEY"] = userdata.get("PINECONE_API_KEY")

    pc = Pinecone()
    if index_name == "all":
        for index in pc.list_indexes().names():
            pc.delete_index(index)
    else:
        pc.delete_index(index_name)
    print(f"deleted {index_name}")


def ask_question(query, vector_store):
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA

    template = """
  use the following pieces of context to answer the question at the end. if you don't the answer just say that you don't know the answer, don't try to make up an answer, keep the answer as concise as possible
  {context}
  Question:{question}
  """
    QA_CHAIN_TEMPLATE = PromptTemplate.from_template(template)
    pinecone_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=1),  # type: ignore
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_TEMPLATE},
        verbose=True,
    )

    response = pinecone_chain({"query": query})
    return response


def searching_with_custom_prompt(query, vector_store, search_type="llm"):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_google_genai import GoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
    import logging

    logging.basicConfig(level=logging.DEBUG)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=FileChatMessageHistory("chat_history.json"),
        input_key="question",
        output_key="answer",
    )

    system_message_prompt = """
    use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: ```{context}```
    """

    user_message_prompt = """
    Question: ```{question}```
    Chat History: ```{chat_history}```
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_message_prompt),
        HumanMessagePromptTemplate.from_template(user_message_prompt),
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)
    llm = GoogleGenerativeAI(model="gemini-pro")  # type: ignore
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )

    logging.debug("Invoking chain with query: %s", query)
    response = chain.invoke({"question": query})
    logging.debug("Chain response: %s", response)

    return response


def config_bot():
    try:
        get_api_keys()
        os.makedirs("files/", exist_ok=True)
        urls = [
            "https://fsciences.univ-setif.dz/main_page/english",
        ]
        mass_download(urls)
        docs = load_docs_locally()
        print(len(docs))
        chunks = chunk_data(docs)
        print(f"{len(chunks)} chunk")
        delete_index()
        vector_store = insert_or_create_index("test-index", chunks)
        pprint(vector_store)
        return True,chunks
    except Exception as e:
        print(e)
        return False,None
