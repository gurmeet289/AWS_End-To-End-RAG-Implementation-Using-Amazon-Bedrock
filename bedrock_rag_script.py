import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Define the prompt template for the question-answering system
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

# Initialize AWS Bedrock client using boto3
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Instantiate the embedding model for generating text embeddings using Bedrock
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def get_documents():
    """
    Loads and splits PDF documents from the 'data' directory into chunks.

    Returns:
        docs (list): A list of text chunks from the PDF documents.
    """
    loader = PyPDFDirectoryLoader("data")  # Loads PDF files from the 'data' directory
    documents = loader.load()
    
    # Splitting text into chunks of 1000 characters with 500 characters overlapping for context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    
    return docs


def get_vector_store(docs):
    """
    Creates a FAISS vector store from the documents and saves it locally.

    Args:
        docs (list): A list of document chunks.
    """
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")  # Save the FAISS index locally


def get_llm():
    """
    Initializes and returns the Bedrock LLM (large language model) with the necessary configurations.

    Returns:
        llm (Bedrock): The initialized LLM object.
    """
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    
    return llm


# Define the prompt template used in the RetrievalQA chain
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def get_response_llm(llm, vectorstore_faiss, query):
    """
    Retrieves a response from the LLM using the FAISS vector store.

    Args:
        llm (Bedrock): The LLM for answering queries.
        vectorstore_faiss (FAISS): The FAISS vector store for document retrieval.
        query (str): The user's question/query.

    Returns:
        str: The generated answer.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Type of chain for document retrieval
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Get the answer along with source documents
    answer = qa({"query": query})
    return answer['result']


def main():
    """
    Streamlit app entry point to manage the user interface for the RAG demo.

    The application allows users to input questions, create or update the vector store, and retrieve answers from the LLM.
    """
    st.set_page_config("RAG Demo")  # Set the title of the Streamlit page
    st.header("End-to-end RAG Application")  # Main header of the app
    user_question = st.text_input("Ask a Question from the PDF Files")  # User input for question

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Store Vector"):
            """
            Button to create or update the vector store by processing the documents.
            """
            with st.spinner("Processing..."):
                docs = get_documents()  # Load and process documents
                get_vector_store(docs)  # Create and store the FAISS vector store
                st.success("Done")  # Notify user when the process is complete
    
    if st.button("Send"):
        """
        Button to query the system with the user's question.
        """
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)  # Load the saved FAISS index
            llm = get_llm()  # Get the LLM instance
            st.write(get_response_llm(llm, faiss_index, user_question))  # Display the generated answer from the LLM


if __name__ == "__main__":
    main()  # Run the Streamlit app when the script is executed
