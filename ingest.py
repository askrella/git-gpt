#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings

# Load environment variables
chunk_size = 500
chunk_overlap = 50

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),

    # Typescript, Javascript
    ".ts": (TextLoader, {"encoding": "utf8"}),
    ".js": (TextLoader, {"encoding": "utf8"}),

    # HTML, CSS
    ".html": (TextLoader, {"encoding": "utf8"}),
    ".css": (TextLoader, {"encoding": "utf8"}),

    # Python
    ".py": (TextLoader, {"encoding": "utf8"}),
    
    # Golang
    ".go": (TextLoader, {"encoding": "utf8"}),
    ".mod": (TextLoader, {"encoding": "utf8"}),

    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.append(doc)
                pbar.update()

    return results

def process_documents(source_directory: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def get_chroma_db(documents_path: str, db_directory: str, embeddings_model_name: str) -> Chroma:
    # Create embeddings
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    print(f"Vectorstore directory: {db_directory}")
    chrome_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=db_directory,
        anonymized_telemetry=False
    )

    if does_vectorstore_exist(db_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {db_directory}")
        db = Chroma(persist_directory=db_directory, embedding_function=embeddings, client_settings=chrome_settings)
        return db
        #collection = db.get()
        #texts = process_documents(documents_path, [metadata['source'] for metadata in collection['metadatas']])
        #print(f"Creating embeddings. May take some minutes...")
        # db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents(documents_path)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=db_directory, client_settings=chrome_settings)
    db.persist()
    
    return db
