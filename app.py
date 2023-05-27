# General modules
import os
import dotenv

# Langchain
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

# Own modules
from util import ask_for_repo, clone_repo, get_repo_name, rm_recursively
from ingest import get_chroma_db

# Load environment variables
dotenv.load_dotenv()

# Constants
db_path = "./db"
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
openai_api_key = os.environ.get('OPENAI_API_KEY')

def use_gpt():
    return openai_api_key is not None and len(openai_api_key) > 0

def start():
    # Create db_path if it doesn't exist
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    
    # Ask for repo
    repo_url = ask_for_repo()
    print("Cloning repo...")
    repo_path = clone_repo(repo_url)

    # Load documents
    print("Loading chroma database...")
    db_name = os.path.join(db_path, get_repo_name(repo_url))
    db: Chroma = get_chroma_db(repo_path, db_name, embeddings_model_name)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Delete repo
    print("Deleting repo...")
    rm_recursively(repo_path)

    # LLM
    llm = None

    if use_gpt():
        print("Using OpenAI API")
        llm = ChatOpenAI(openai_api_key=openai_api_key)
    else:
        print("Using GPT4ALL")
        llm = GPT4All(
            model=model_path,
            n_ctx=model_n_ctx,
            backend="gptj"
        )

    # Retrieval QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    print("")
    print("Ready to answer questions.")
    print("")
    
    while True:
        # Ask for user prompt
        query = input("Enter a question: ")
        
        # Process query
        response = None
        if use_gpt():
            with get_openai_callback() as cb:
                response = qa(query)
                print("------")
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                print("------")
        else:
            response = qa(query)
    
        # Print answer
        answer = response['result']
        print(f"Answer: {answer}")
        

# Main method
if __name__ == "__main__":
    start()
