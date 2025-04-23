import requests
import json
import base64
import os
import sys
import argparse
from pathlib import Path
import re  # Added for word counting


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fetch README files from all repositories of a given GitHub user"
    )
    parser.add_argument("username", help="GitHub username")
    parser.add_argument("output_dir", help="Directory to save README files")
    parser.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="Minimum word count for README files to save (default: 0, save all)",
    )
    parser.add_argument(
        "--token", help="GitHub token (optional if GITHUB_TOKEN env var is set)"
    )
    return parser.parse_args()


def count_words(text):
    """Count the number of words in a text"""
    # Remove Markdown formatting symbols that might be counted as words
    # This simplistic approach removes common Markdown markers
    cleaned_text = re.sub(r"[#*_`~\[\]\(\)\{\}]", " ", text)

    # Split by whitespace and count non-empty elements
    words = [word for word in cleaned_text.split() if word.strip()]
    return len(words)


def get_user_repos(username, token):
    """Fetch all repositories for a given user"""
    all_repos = []
    page = 1

    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching repositories: {response.status_code}")
            break

        repos = response.json()
        if not repos:  # Empty page, we've reached the end
            break

        all_repos.extend(repos)
        page += 1

    return all_repos


def get_readme(username, repo_name, token):
    """Fetch the README file for a specific repository"""
    url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"No README found for {repo_name}")
        return None

    readme_data = response.json()
    # GitHub returns base64 encoded content
    content = base64.b64decode(readme_data["content"]).decode("utf-8")
    return content


def main():
    args = parse_arguments()

    # Get GitHub token from args or environment
    github_token = args.token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print(
            "Please provide a GitHub token using --token or set the GITHUB_TOKEN environment variable"
        )
        sys.exit(1)

    username = args.username
    output_dir = args.output_dir
    min_words = args.min_words

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all repositories for the user
    print(f"Fetching repositories for {username}...")
    repos = get_user_repos(username, github_token)
    print(f"Found {len(repos)} repositories")

    # Track stats
    total_readmes = 0
    saved_readmes = 0

    # For each repository, fetch and save the README if it meets the minimum word count
    for repo in repos:
        repo_name = repo["name"]
        print(f"Processing repository: {repo_name}")

        readme = get_readme(username, repo_name, github_token)
        if readme:
            total_readmes += 1

            # Check if README meets minimum word count requirement
            word_count = count_words(readme)
            if word_count >= min_words:
                with open(
                    f"{output_dir}/{repo_name}_README.md", "w", encoding="utf-8"
                ) as f:
                    f.write(readme)
                saved_readmes += 1
                print(f"Saved README for {repo_name} ({word_count} words)")
            else:
                print(
                    f"Skipped README for {repo_name} - too short ({word_count} < {min_words} words)"
                )

    print(f"\nSummary:")
    print(f"- Total repositories: {len(repos)}")
    print(f"- READMEs found: {total_readmes}")
    print(f"- READMEs saved (≥ {min_words} words): {saved_readmes}")
    print(f"- READMEs skipped (< {min_words} words): {total_readmes - saved_readmes}")
    print(f"\nAll qualifying README files have been saved to {output_dir}")


if __name__ == "__main__":
    main()
