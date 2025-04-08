#https://python.langchain.com/docs/tutorials/qa_chat_history/

from langchain_core import documents
#from langchain_ollama import ChatOllama
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
#from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain.embeddings import OllamaEmbeddings
import os
from typing import Annotated, Literal, Sequence, List
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
#from langchain.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import MemorySaver
import pdfplumber
import streamlit as st
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from htmlTemplates import css, bot_template, user_template
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
#from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import SentenceTransformerEmbeddings

import os

load_dotenv()

api_key= os.getenv('GROQ_API_KEY')
google_key=os.getenv('GOOGLE_API_KEY')
#local_llm = 'llama3.2:3b-instruct-fp16'
model_name = 'llama-3.1-70b-versatile'
#llm = ChatOllama(model=local_llm, temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name
    )

#llm = groq_chat

#embeddings= OllamaEmbeddings()
#embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#embeddings=OllamaEmbeddings(
#  model='mxbai-embed-large',
#)
#embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


#Load documents
def load_db(embeddings, path):
    text =''

    #with open(path,'rb') as file:
    #    pdf_reader = PdfReader(file)
    #    for page in pdf_reader.pages:
    #        text += page.extract_text()
    #        print(text)
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    #text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

if not os.path.exists('faiss_index'):
    vectorstore=load_db(embeddings,'s00484-022-02405-z.pdf')
    vectorstore.save_local("faiss_index")
else:
    vectorstore = FAISS.load_local("faiss_index",embeddings=embeddings,allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()
# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an expert professor for checking answers of students. You do not have to answer questions about the retrieved document."
    "The students will write a question about the document and their answer. You have to check whether their answer is correct."  
    "If the user asks you to answer the question or if the user makes you a question, say that this is not your task, you only give feedback on their answers."
    "Answer always in spanish, please."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import AIMessage, HumanMessage

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


# Our graph consists only of one node:
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Finally, we compile the graph with a checkpointer object.
# This persists the state, in this case in memory.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


config = {"configurable": {"user_id": "1", "thread_id": "1"}}
text_contents = "Tu chat:"
save_chat = ""
st.header('Chatbot CADI ')
st.write(bot_template.replace("{{MSG}}", "Hola, estoy aquí para ayudarte, ¿Cómo te llamas?"), unsafe_allow_html=True)
question = st.chat_input("Escribe la pregunta y tu respuesta")
if question:
    text_contents=text_contents+"Tu:"+question+"\n"
    result=app.invoke({"input": question},config)
    text_contents=text_contents+"Bot:"+result['answer']+"\n"
    st.write(css, unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", result["answer"]), unsafe_allow_html=True)
    st.session_state.chat_history = result['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        print(i, message)

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            #st.write(message.content)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            #st.write(message.content)

save_chat = save_chat+text_contents
st.download_button("Download tu chat", save_chat)
