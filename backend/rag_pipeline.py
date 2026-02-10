import os
import shutil
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# =====================================================
# 🔐 SARVAM CHAT MODEL (NO OpenAI, NO ChatOpenAI)
# =====================================================

class SarvamChatModel(BaseChatModel):
    model: str = "sarvam-m"

    def _generate(self, messages, stop=None, **kwargs):
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY not set")

        payload = {
            "model": self.model,
            "messages": [
                {"role": m.type, "content": m.content}
                for m in messages
            ]
        }

        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(response.text)

        text = response.json()["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    @property
    def _llm_type(self):
        return "sarvam-chat"


# =====================================================
# 🧹 HTML CLEANING
# =====================================================

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# =====================================================
# 🌐 DATA SOURCES
# =====================================================

important_urls = [
    "https://www.vvitguntur.com/about/about-us",
    "https://www.vvitguntur.com/about/keypersons",
    "https://www.vvitguntur.com/about/vision",
    "https://www.vvitguntur.com/about/people",
    "https://www.vvitguntur.com/departments/cse-home",
    "https://www.vvitguntur.com/cse-faculty",
    "https://www.vvitguntur.com/placements-home/year-wise-placements",
    "https://www.vvitguntur.com/placements-home/placement-recruiters",
]

PERSIST_DIRECTORY = "./chroma_db"


# =====================================================
# 📦 EMBEDDINGS + VECTORSTORE
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = []
needs_rebuild = not os.path.exists(PERSIST_DIRECTORY)

if needs_rebuild:
    for url in important_urls:
        loader = WebBaseLoader(url)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.page_content = clean_html(doc.page_content)
            doc.metadata["source"] = url
        docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectorstore.persist()
else:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )


retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


# =====================================================
# 🧠 PROMPT
# =====================================================

prompt = PromptTemplate(
    template="""
You are CollegeBot, a friendly assistant for
Vasireddy Venkatadri Institute of Technology (VVIT), Guntur.

Answer clearly and confidently.
Do not mention documents, context, or sources.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# =====================================================
# 🤖 LLM + QA CHAIN
# =====================================================

llm = SarvamChatModel()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


# =====================================================
# 🧩 QUERY NORMALIZATION
# =====================================================

def normalize_query(q: str) -> str:
    q = q.lower().strip()
    q = q.replace("chair person", "chairman")
    q = q.replace("head of college", "principal")
    return q


# =====================================================
# 🚀 MAIN FUNCTION
# =====================================================

def answer_question(question: str):
    try:
        query = normalize_query(question)
        response = qa_chain.invoke({"query": query})

        return {
            "status": "success",
            "answer": response["result"],
            "sources": len(response["source_documents"])
        }
    except Exception as e:
        return {
            "status": "error",
            "answer": str(e),
            "sources": 0
        }


# =====================================================
# 🧪 LOCAL TEST
# =====================================================

if __name__ == "__main__":
    print(answer_question("Who is the principal of VVIT?"))
