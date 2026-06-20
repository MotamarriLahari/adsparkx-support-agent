from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

def ingest_documents():
    print("📂 Loading documents from /data ...")

    # Load .txt files
    txt_loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()

    # Load .pdf files
    pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()

    docs = txt_docs + pdf_docs
    print(f"✅ Loaded {len(docs)} raw documents")

    # Chunk documents
    print("✂️  Splitting documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    # Generate embeddings + store in ChromaDB
    print("🧠 Generating embeddings and storing in ChromaDB ...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    db.persist()

    print(f"🎉 Done! {len(chunks)} chunks stored in ./chroma_db")

if __name__ == "__main__":
    ingest_documents()