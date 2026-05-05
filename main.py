import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Loading the environment variables from .env file
load_dotenv()

def build_qa_agent(pdf_path, query):
    print("1. Ingesting and Splitting Document...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Chunking: 1000 characters per chunk, 200 character overlap to maintain context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("2. Setting up Vector Database...")
    # Using persist_directory saves the embeddings locally so we don't pay OpenAI every time
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db" 
    )
    # k=3 means retrieving the top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("3. Initializing LLM and Agentic Workflow...")
    # Temperature 0 ensures factual, deterministic answers (crucial for clinical data)
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

    system_prompt = (
        "You are an autonomous clinical research agent. Use the retrieved context "
        "to answer the question. If you do not know the answer, state that you do not know. "
        "Do not hallucinate external information."
        "\n\nContext: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    print("4. Executing Chain...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]

if __name__ == "__main__":
    # Ensuring we have a file named 'sample_document.pdf' in the root folder
    target_pdf = "sample_document.pdf"
    
    # Test Query 1: Factual Retrieval
    test_query = "Why is Smoking bad?"
    
    print(f"\n--- Processing Query: '{test_query}' ---")
    try:
        answer = build_qa_agent(target_pdf, test_query)
        print("\nAGENT RESPONSE:")
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")