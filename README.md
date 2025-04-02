# AWS_End-To-End-RAG-Implementation-Using-Amazon-Bedrock

## 🔑 AWS Credentials Setup

### Before running the application, you need to configure AWS credentials for accessing Bedrock API.

#### Method 1: Using AWS CLI (Recommended)
Install AWS CLI if not already installed:
      
      pip install awscli

Run the following command:
  
      aws configure

Enter your credentials when prompted:
      
      AWS Access Key ID: YOUR_ACCESS_KEY
      AWS Secret Access Key: YOUR_SECRET_KEY
      Default region name: us-east-1

#### Method 2: Using Environment Variables
  
Alternatively, set credentials in your terminal:
    
      export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
      export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
      export AWS_REGION=us-east-1

For Windows (Command Prompt):
    
      set AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
      set AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
      set AWS_REGION=us-east-1

#### Note: Ensure you have the correct IAM permissions for Bedrock API access.

      1) Make sure your AWS credentials are set up to use Bedrock API.
      2) Ensure PDF files are placed inside the data/ folder for the RAG app.
      3) For FAISS vector search, existing indexes are stored locally in faiss_index/.


# Bedrock Chatbot and RAG Application  

This repository contains two implementations leveraging **AWS Bedrock**, **LangChain**, and **Streamlit**:  
1. **RAG (Retrieval-Augmented Generation) Application** – A chatbot that retrieves answers from PDFs.  
2. **Bedrock Chatbot** – A language-specific chatbot that responds to user queries.  

## Features  

1) Uses **AWS Bedrock** for LLM processing  
2) Integrates **LangChain** for prompt handling and retrieval  
3) Implements **FAISS** for vector storage and retrieval  
4) Streamlit UI for user interaction  

---

## 1) RAG (Retrieval-Augmented Generation) Application  

This chatbot processes **PDF files**, converts them into embeddings, and retrieves the most relevant information before responding.  

### How It Works  

1. Loads **PDF documents** from the `data/` folder.  
2. Splits text into **chunks** for efficient retrieval.  
3. Uses **FAISS** for storing embeddings.  
4. Uses **AWS Bedrock LLM** (Meta Llama 2 70B) to generate responses.  

### Installation  

      git clone https://github.com/your-username/your-repo-name.git
      cd your-repo-name
      pip install -r requirements.txt

### Run the Application

      streamlit run your_rag_app_name.py

## 2) Bedrock Chatbot

      This is a chatbot that allows users to interact in different languages (English, Spanish, Hindi, French).

### How It Works

      User selects a language from the sidebar.

### Enters a question.

      The chatbot generates a context-aware response.

### Run the Chatbot

      streamlit run bedrock_rag_script.py

### Requirements

      Python 3.8+
      AWS credentials configured (boto3)
      Streamlit
      LangChain
      FAISS

### Install Dependencies

      pip install -r requirements.txt
