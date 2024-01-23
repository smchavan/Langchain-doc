import os
from langchain_community.llms import OpenAI
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# Function to load variables from .env file
def load_env(file_path=".env"):
    with open(file_path) as f:
        for line in f:
            # Ignore comments and empty lines
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Load environment variables from .env
load_env()
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# Load the Word documents
doc1_path = "/Users/supriya/AI/test-langchain/Data/Test-Data1.docx"
doc2_path = "/Users/supriya/AI/test-langchain/Data/Test-Data2.docx"

loader = Docx2txtLoader(doc1_path)
doc1 = loader.load()

loader = Docx2txtLoader(doc2_path)
doc2 = loader.load()


# Extract the text from the documents
doc1_text = doc1.page_content
doc2_text = doc2.page_content
# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
doc1_chunks = text_splitter.split_text(doc1_text)
doc2_chunks = text_splitter.split_text(doc2_text)

# Create embeddings for the document chunks
embeddings = OpenAIEmbeddings()
doc1_embeddings = [embeddings.embed(chunk) for chunk in doc1_chunks]
doc2_embeddings = [embeddings.embed(chunk) for chunk in doc2_chunks]

# Compare the embeddings
retriever = FAISS.from_embeddings(doc1_embeddings)
qa_chain = RetrievalQA.from_retriever(retriever)
differences = []
for i, emb in enumerate(doc2_embeddings):
    result = qa_chain({"question": emb})
    if result["result"] != "Match":
        differences.append((i, result["result"]))

# Display the differences
for diff in differences:
    chunk_index, difference = diff
    print(f"Difference found in chunk {chunk_index}: {difference}")

