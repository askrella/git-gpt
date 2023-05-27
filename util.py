from git import Repo
import os
import stat
import tempfile
import uuid

def ask_for_repo():
    repo_url = input("Enter repo url: ")
    return repo_url

def get_repo_name(repo_url: str):
    return repo_url.split("/")[-1].split(".")[0]

def clone_repo(repo_url: str):
    # Clone repo
    repo_tmp_path = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    Repo.clone_from(repo_url, repo_tmp_path)
    return repo_tmp_path

def get_repo_name(repo_url: str):
    return repo_url.split("/")[-1].split(".")[0]

def rm_recursively(repo_tmp_path: str):
    for root, dirs, files in os.walk(repo_tmp_path, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(repo_tmp_path)      
