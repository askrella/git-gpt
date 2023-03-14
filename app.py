import os
import uuid
import stat
import datetime
import openai
import inquirer
import dotenv
import temp
from git import Repo
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# Contants
STORE_DIR = "store"
CLONE_NEW_REPO = "Clone new repo"
LOAD_EXISTING_INDEX = "Load existing index"

def ask_for_repo():
    repo_url = input("Enter repo url: ")
    return repo_url

def clone_repo(repo_url):
    repo_tmp_path = os.path.join(temp.tempdir(), uuid.uuid4().hex)
    Repo.clone_from(repo_url, repo_tmp_path)
    return repo_tmp_path

def get_repo_name(repo_url):
    return repo_url.split("/")[-1].split(".")[0]

def rm_recursively(repo_tmp_path):
    for root, dirs, files in os.walk(repo_tmp_path, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(repo_tmp_path)      

def tokens_to_cost(tokens):
    return (tokens / 1000) * 0.02

def save_index(index, repo_url):
    # Create store/ dir if it doesn't exist
    if not os.path.exists(STORE_DIR):
        os.makedirs(STORE_DIR)

    # Save index
    date_str = datetime.datetime.now().strftime("%m-%H-%d-%m-%Y")
    name = get_repo_name(repo_url) + "-" + date_str + ".json"
    index.save_to_disk(os.path.join(STORE_DIR, name))

def start_asking(index):
    while True:
        try:
            # Ask for query
            query = input("Enter query: ")
            response = index.query(query)
            print(response)
        except KeyboardInterrupt:
            break   

def new_repo_flow():
    # Ask for repo & clone it
    repo_url = ask_for_repo()
    print("Cloning repo...")
    repo_tmp_path = clone_repo(repo_url)

    # Load documents & create the index
    print("Creating index...")
    documents = SimpleDirectoryReader(input_dir=repo_tmp_path, exclude_hidden=True).load_data()
    index = GPTSimpleVectorIndex(documents)

    # Delete repo
    rm_recursively(repo_tmp_path)

    # Save index
    save_index(index, repo_url)

    return index

def existing_repo_flow():
    # Show user list of existing indexes
    print("Existing indexes:")
    for file in os.listdir(STORE_DIR):
        print(file)

    # Ask user to select index
    files_array = []

    for file in os.listdir(STORE_DIR):
        files_array.append(file)

    answer = inquirer.prompt(
        [inquirer.List("index", message="Select index", choices=files_array)]
    )
    index_path = os.path.join(STORE_DIR, answer["index"])

    # Load index
    index = GPTSimpleVectorIndex.load_from_disk(index_path)

    return index

def main():
    # Declare index
    index = None

    # Load .env
    dotenv.load_dotenv()

    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY env is not set")
        return
    
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Print welcome message    
    print("")
    print("Welcome to Git GPT!")
    print("")

    # Ask if user wants to clone new repo or load existing index
    answer = inquirer.prompt(
        [inquirer.List("type", message="Would you like to clone a new repo or load an existing index?", choices=[
            CLONE_NEW_REPO, LOAD_EXISTING_INDEX
        ])]
    )["type"]

    # Handle user input
    if answer == CLONE_NEW_REPO:
        # New repo, new index
        index = new_repo_flow()
    elif answer == LOAD_EXISTING_INDEX:
        # Existing index
        index = existing_repo_flow()
    else:
        print("Invalid input, exiting...")
        return

    # Validate index
    if index is None:
        print("Failed to initialize index")
        return

    # Start asking for queries
    start_asking(index)

if __name__ == "__main__":
    main()