import requests
import json
import os
from datetime import datetime, timedelta
from src import DATA_DIR
import argparse
import time
import base64


# Constants
DATA_DIR = DATA_DIR+"python_repos_may_2024"
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_AT") 

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def search_repos(min_stars, max_size_kb, language="python", date=None):
    search_url = f"{GITHUB_API_URL}/search/repositories"
    query = f"language:{language} stars:>={min_stars} size:<={max_size_kb}"
    
    if date:
        query += f" created:{date}"
    
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": 100
    }
    
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()['items']
    else:
        print(f"Search failed: {response.status_code}")
        return []

def has_long_python_file(repo_full_name, min_lines):
    tree_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/git/trees/main?recursive=1"
    response = requests.get(tree_url, headers=headers)
    if response.status_code == 200:
        tree = response.json()
        python_files = [file for file in tree['tree'] if file['path'].endswith('.py') and file['size'] > min_lines * 40]  # Estimate 30 bytes per line
        if python_files:
            file_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{python_files[0]['path']}"
            file_response = requests.get(file_url, headers=headers)
            if file_response.status_code == 200:
                content = base64.b64decode(file_response.json()['content']).decode('utf-8')
                return len(content.splitlines()) > min_lines
    return False

def clone_repo(repo_full_name):
    clone_url = f"https://github.com/{repo_full_name}.git"
    repo_dir = os.path.join(DATA_DIR, repo_full_name.split('/')[-1])
    os.system(f"git clone {clone_url} {repo_dir}")
    print(f"Cloned: {repo_full_name}")

def get_next_date(current_date):
    return (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

def main(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    downloaded_count = 0
    current_date = args.start_date

    while downloaded_count < args.num_repos:
        print(f"Searching repositories created on {current_date}")
        repos = search_repos(args.min_stars, args.max_size * 1024, date=current_date)
        
        for repo in repos:
            if has_long_python_file(repo['full_name'], args.min_lines):
                clone_repo(repo['full_name'])
                downloaded_count += 1
                if downloaded_count == args.num_repos:
                    break
        
        if downloaded_count < args.num_repos:
            current_date = get_next_date(current_date)
        
        time.sleep(2)

    print(f"Downloaded {downloaded_count} Python repositories to {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Python repositories from GitHub")
    parser.add_argument("--max-size", type=int, default=25, help="Maximum repository size in MB")
    parser.add_argument("--min-stars", type=int, default=10, help="Minimum number of stars")
    parser.add_argument("--min-lines", type=int, default=100, help="Minimum lines in a Python file")
    parser.add_argument("--num-repos", type=int, default=10, help="Number of repositories to download")
    parser.add_argument("--start-date", type=str, default="2024-05-01", help="Start date for repository search (YYYY-MM-DD)")
    args = parser.parse_args()

    main(args)