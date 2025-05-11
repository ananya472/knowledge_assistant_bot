import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load Hugging Face Token
HF_TOKEN = os.environ.get("HF_TOKEN")  # Ensure this environment variable is set
if not HF_TOKEN:
    raise ValueError("Hugging Face API token is missing! Set HF_TOKEN as an environment variable.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Setup LLM (Mistral with Hugging Face)
def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        token=HF_TOKEN,  
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
        task="text-generation",
        model_kwargs={ 
            "max_length": 256  
        }
    )

# Step 2: Define Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
        You are an AI expert assistant for **TechNova Solutions**, trained on official documents including the company profile, product specifications, and FAQs.  
You provide **accurate, structured, and concise** answers to user questions about the company and its offerings.

## **User Question**:
{question}

## **Context (for reference)**:
{context}

## **Instructions for Response**:
- **Answer only if the question is related to TechNova, its products, or services**.  
- **Use bullet points** or **numbered lists** where appropriate.  
- Keep answers **clear, to the point, and non-repetitive**.  
- Refer directly to content from the context and avoid hallucinating facts.  
- If the question is **not related to TechNova**, respond:  
  _"Sorry, I can only answer questions related to TechNova Solutions and its services."_  
- Maintain a **professional and helpful tone**.  

---

**🌐 Example Response Format**

**✅ About TechNova Services**  
- **Cloud Computing**: Offers secure, scalable infrastructure for enterprise-grade deployments.  
- **AI Solutions**: Includes tools for ML model training, deployment, and monitoring.  
- **Data Analytics**: Real-time dashboards and insights to support business decisions.  

---

**✅ Product Capabilities – NovaAI Cloud Suite**  
1️⃣ **Machine Learning Tools**  
- Built-in models for regression, classification, and forecasting.  
- Supports frameworks like TensorFlow and PyTorch.  

2️⃣ **Security Features**  
- AES-256 encryption, MFA, and role-based access control.  
- Compliance with GDPR and HIPAA.

---

**✅ Frequently Asked Questions (FAQs)**  
**Q: Does TechNova support third-party integrations?**  
- Yes, NovaAI Cloud Suite integrates with tools like Salesforce, Google Cloud, and AWS.  
- Offers REST APIs for custom connectors.

---

**Now, provide your response below following this structure.**

"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 3: Load FAISS Database
DB_FAISS_PATH = r"C:\Users\anany\Desktop\CHATBOT\vectorstore\db_faiss"

# ✅ **Fix: Use the correct way to load the embedding model**
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print("Error loading embedding model:", str(e))
    raise

# ✅ **Fix: Ensure FAISS is loaded properly**
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print("Error loading FAISS database:", str(e))
    raise

# Step 4: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Get User Query
user_query = input("Write Query Here: ")

# Step 6: Get Response
try:
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response.get("result", "No result found."))
    print("SOURCE DOCUMENTS: ", response.get("source_documents", []))
except Exception as e:
    print("Error during query processing:", str(e))

