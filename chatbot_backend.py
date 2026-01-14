
from __future__ import annotations
from langgraph.graph import StateGraph,START,END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
from langgraph.checkpoint.memory import MemorySaver
import tempfile
import requests
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Dict,Any,Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from datetime import datetime
import csv
import io



# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None



def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
     """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
     if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

     try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
      )
        chunks = splitter.split_documents(docs)
        embeddings=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
     finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass





search_tool=DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def get_weather(city: str):
    """
    fetches the current wheather of any place 
    by using a wheather api provided
    """

    try:
        api_key="f3c44a65a12c8ad9ea7f8217ce1260ed"
        url= f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        response =requests.get(url)

        return response.json()

    except requests.exceptions.RequestException as e:
        return{"error": str(e)}
    
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }




import requests
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any




# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
github_token = os.getenv("GITHUB_TOKEN")



def get_github_headers():
    """Get GitHub API headers with authentication"""
    
    return {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

def clean_github_input(value: str) -> str:
    """Clean and validate input parameters"""
    if not value:
        return value
    cleaned = value.strip().strip('"').strip("'").strip()
    cleaned = cleaned.replace('\n', '').replace('\r', '')
    return cleaned

def make_github_request(method: str, endpoint: str, data=None, params=None):
    """Make authenticated GitHub API request with proper token handling"""
    
    # Check if token exists
    if not github_token:
        raise Exception("GITHUB_TOKEN not found in environment variables. Please set it in your .env file.")
    
    # Use 'token' prefix for classic PATs (most common)
    # GitHub also accepts 'Bearer' for fine-grained tokens
    headers = {
        "Authorization": f"token {github_token}",  # Changed from "Bearer" to "token"
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    url = f"{GITHUB_API_BASE}{endpoint}"
    
    print(f"[DEBUG] {method} {url}")
    if data:
        print(f"[DEBUG] Request data keys: {list(data.keys()) if isinstance(data, dict) else 'non-dict data'}")
    
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,  # This automatically sets Content-Type: application/json
            params=params,
            timeout=30  # Add timeout to prevent hanging
        )
        
        print(f"[DEBUG] Response status: {response.status_code}")
        
        # Handle DELETE requests that return 204 No Content
        if response.status_code == 204:
            return {"status": "success", "message": "Operation completed successfully"}
        
        # Check if request was successful
        if response.status_code not in [200, 201, 204]:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", "Unknown error")
                
                # Add more context for authentication errors
                if response.status_code == 401:
                    error_msg = f"Authentication failed: {error_msg}. Check if your GITHUB_TOKEN is valid and not expired."
                elif response.status_code == 403:
                    error_msg = f"Access forbidden: {error_msg}. Your token may lack required permissions (needs 'repo' or 'public_repo' scope)."
                elif response.status_code == 404:
                    error_msg = f"Not found: {error_msg}. Check if the repository/user exists and you have access."
                
            except:
                error_msg = response.text or "Unknown error"
            
            raise Exception(f"GitHub API Error ({response.status_code}): {error_msg}")
        
        # Return parsed JSON
        return response.json()
        
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Connection error. Check your internet connection.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")



# =============================================================================
# REPOSITORY MANAGEMENT TOOLS
# =============================================================================

@tool
def github_create_repository(
    name: str, 
    description: str = "", 
    private: bool = False, 
    auto_init: bool = True
) -> dict:
    """
    Create a new GitHub repository in your account
    
    Args:
        name: Repository name (required)
        description: Repository description
        private: Create private repo (default: False - public)
        auto_init: Initialize with README (default: True)
    
    Returns:
        Dictionary with repository information including clone_url, html_url
    """
    print(f"[TOOL] GitHub: Creating repository '{name}'")
    
    data = {
        "name": clean_github_input(name),
        "description": description,
        "private": private,
        "auto_init": auto_init
    }
    
    try:
        result = make_github_request("POST", "/user/repos", data=data)
        return {
            "status": "success",
            "message": f"Repository '{name}' created successfully",
            "name": result.get("name"),
            "full_name": result.get("full_name"),
            "html_url": result.get("html_url"),
            "clone_url": result.get("clone_url"),
            "private": result.get("private"),
            "description": result.get("description")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@tool
def github_delete_repository(owner: str, repo: str) -> dict:
    """
    Delete a GitHub repository (WARNING: Permanent!)
    
    Args:
        owner: Repository owner (your GitHub username)
        repo: Repository name
    
    Returns:
        Dictionary with deletion status
    """
    print(f"[TOOL] GitHub: Deleting repository {owner}/{repo}")
    
    owner = clean_github_input(owner)
    repo = clean_github_input(repo)
    
    try:
        result = make_github_request("DELETE", f"/repos/{owner}/{repo}")
        
        # DELETE returns 204 No Content on success
        return {
            "status": "success",
            "message": f"Repository {owner}/{repo} deleted successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete repository: {str(e)}"
        }


@tool
def github_update_repository(
    owner: str, 
    repo: str, 
    new_name: str = None,
    description: str = None, 
    private: bool = None
) -> dict:
    """
    Update GitHub repository settings
    
    Args:
        owner: Repository owner
        repo: Current repository name
        new_name: New repository name (optional)
        description: New description (optional)
        private: Make private/public (optional)
    
    Returns:
        Dictionary with updated repository information
    """
    print(f"[TOOL] GitHub: Updating repository {owner}/{repo}")
    
    owner = clean_github_input(owner)
    repo = clean_github_input(repo)
    
    data = {}
    if new_name:
        data["name"] = clean_github_input(new_name)
    if description is not None:
        data["description"] = description
    if private is not None:
        data["private"] = private
    
    if not data:
        return {
            "status": "error",
            "message": "No updates provided. Specify at least one parameter to update."
        }
    
    try:
        result = make_github_request("PATCH", f"/repos/{owner}/{repo}", data=data)
        return {
            "status": "success",
            "message": f"Repository {owner}/{repo} updated successfully",
            "name": result.get("name"),
            "full_name": result.get("full_name"),
            "html_url": result.get("html_url"),
            "description": result.get("description"),
            "private": result.get("private")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update repository: {str(e)}"
        }


@tool
def github_get_repository(owner: str, repo: str) -> dict:
    """
    Get detailed information about a GitHub repository
    
    Args:
        owner: Repository owner
        repo: Repository name
    
    Returns:
        Dictionary with detailed repository information
    """
    print(f"[TOOL] GitHub: Getting repository info for {owner}/{repo}")
    
    owner = clean_github_input(owner)
    repo = clean_github_input(repo)
    
    try:
        result = make_github_request("GET", f"/repos/{owner}/{repo}")
        return {
            "status": "success",
            "name": result.get("name"),
            "full_name": result.get("full_name"),
            "description": result.get("description"),
            "html_url": result.get("html_url"),
            "clone_url": result.get("clone_url"),
            "private": result.get("private"),
            "fork": result.get("fork"),
            "created_at": result.get("created_at"),
            "updated_at": result.get("updated_at"),
            "pushed_at": result.get("pushed_at"),
            "size": result.get("size"),
            "stargazers_count": result.get("stargazers_count"),
            "watchers_count": result.get("watchers_count"),
            "forks_count": result.get("forks_count"),
            "open_issues_count": result.get("open_issues_count"),
            "language": result.get("language"),
            "default_branch": result.get("default_branch"),
            "topics": result.get("topics", [])
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get repository info: {str(e)}"
        }


@tool
def github_list_user_repositories(username: str = None, type: str = "all", max_results: int = 30) -> dict:
    """
    List repositories for a user
    
    Args:
        username: GitHub username (None for authenticated user)
        type: Repository type (all, owner, member) - only applies to authenticated user
        max_results: Maximum number of repositories to return
    
    Returns:
        Dictionary with list of repositories
    """
    if username:
        print(f"[TOOL] GitHub: Listing repositories for user '{username}'")
        endpoint = f"/users/{username}/repos"
        params = {"per_page": max_results}
    else:
        print(f"[TOOL] GitHub: Listing repositories for authenticated user")
        endpoint = "/user/repos"
        params = {"type": type, "per_page": max_results}
    
    try:
        result = make_github_request("GET", endpoint, params=params)
        
        repos = []
        for repo in result:
            repos.append({
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "html_url": repo.get("html_url"),
                "private": repo.get("private"),
                "fork": repo.get("fork"),
                "language": repo.get("language"),
                "stargazers_count": repo.get("stargazers_count"),
                "forks_count": repo.get("forks_count"),
                "updated_at": repo.get("updated_at")
            })
        
        return {
            "status": "success",
            "message": f"Found {len(repos)} repositories",
            "count": len(repos),
            "repositories": repos
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list repositories: {str(e)}",
            "repositories": []
        }


# =============================================================================
# FILE CREATION TOOL
# =============================================================================

@tool
def github_create_file(
    owner: str, 
    repo: str, 
    path: str, 
    content: str,
    commit_message: str,
    branch: str = "main"
) -> dict:
    """
    Create a new file in a GitHub repository
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: File path (e.g., 'src/main.py' or 'README.md')
        content: File content as string
        commit_message: Commit message for this creation
        branch: Branch name (default: main)
    
    Returns:
        Dictionary with status and file information
    """
    print(f"[TOOL] GitHub: Creating file {path} in {owner}/{repo}")
    
    owner = clean_github_input(owner)
    repo = clean_github_input(repo)
    # Don't clean the path - it might have necessary characters
    encoded_path = quote(path, safe='/')
    
    # Check if file already exists (this will throw exception if not found)
    try:
        existing_file = make_github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{encoded_path}",
            params={"ref": branch}
        )
        
        # If we get here, file exists
        return {
            "status": "error",
            "message": f"File '{path}' already exists in branch '{branch}'",
            "existing_sha": existing_file.get("sha"),
            "existing_url": existing_file.get("html_url"),
            "suggestion": "Use github_update_file or github_verify_and_update to modify it"
        }
    except Exception as e:
        # File doesn't exist - this is good, we can create it
        print(f"[DEBUG] File doesn't exist (expected): {str(e)[:100]}")
        pass
    
    # Encode content to base64
    try:
        content_bytes = content.encode('utf-8')
        content_base64 = base64.b64encode(content_bytes).decode('utf-8')
        print(f"[DEBUG] Content encoded to base64 (length: {len(content_base64)})")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to encode content: {str(e)}"
        }
    
    # Prepare request body (no SHA needed for creation)
    data = {
        "message": commit_message,
        "content": content_base64,
        "branch": branch
    }
    
    print(f"[DEBUG] Creating file with branch={branch}")
    
    try:
        # Use data= not json= to match your make_github_request signature
        response = make_github_request(
            "PUT", 
            f"/repos/{owner}/{repo}/contents/{encoded_path}", 
            data=data
        )
        
        # Verify success
        if response.get("content") and response.get("commit"):
            return {
                "status": "success",
                "message": f"File '{path}' created successfully",
                "name": response["content"].get("name"),
                "path": response["content"].get("path"),
                "url": response["content"].get("html_url"),
                "sha": response["content"].get("sha"),
                "commit": {
                    "sha": response["commit"].get("sha"),
                    "url": response["commit"].get("html_url")
                }
            }
        else:
            return {
                "status": "warning",
                "message": "File created but response structure unexpected",
                "response": response
            }
            
    except Exception as e:
        error_msg = str(e)
        
        # Parse specific errors
        if "422" in error_msg or "Invalid request" in error_msg:
            return {
                "status": "error",
                "message": f"Invalid request - check if repository exists and you have write access: {error_msg}"
            }
        elif "404" in error_msg:
            return {
                "status": "error",
                "message": f"Repository '{owner}/{repo}' not found or branch '{branch}' doesn't exist"
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to create file: {error_msg}"
            }


# =============================================================================
# SEARCH TOOLS
# =============================================================================

@tool
def github_search_repositories(
    query: str, 
    sort: str = "stars", 
    max_results: int = 10
) -> dict:
    """
    Search GitHub repositories
    
    Args:
        query: Search query (e.g., "language:python stars:>1000")
        sort: Sort by (stars, forks, updated, help-wanted-issues)
        max_results: Maximum results to return (max 100)
    
    Returns:
        Dictionary with search results
    
    Query examples:
        - "machine learning language:python"
        - "stars:>1000 language:javascript"
        - "topic:react topic:typescript"
    """
    print(f"[TOOL] GitHub: Searching repositories with query '{query}'")
    
    params = {
        "q": query, 
        "sort": sort, 
        "per_page": min(max_results, 100)
    }
    
    try:
        result = make_github_request("GET", "/search/repositories", params=params)
        
        items = result.get("items", [])
        repos = []
        
        for repo in items:
            repos.append({
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "html_url": repo.get("html_url"),
                "stargazers_count": repo.get("stargazers_count"),
                "forks_count": repo.get("forks_count"),
                "language": repo.get("language"),
                "topics": repo.get("topics", [])
            })
        
        return {
            "status": "success",
            "message": f"Found {result.get('total_count', 0)} repositories (showing {len(repos)})",
            "total_count": result.get("total_count", 0),
            "repositories": repos
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to search repositories: {str(e)}",
            "repositories": []
        }


@tool
def github_search_code(query: str, max_results: int = 10) -> dict:
    """
    Search code across all GitHub repositories
    
    Args:
        query: Search query (e.g., "language:python def hello")
        max_results: Maximum results to return
    
    Returns:
        Dictionary with code search results
    
    Query examples:
        - "addClass in:file language:js repo:jquery/jquery"
        - "language:python import tensorflow"
        - "filename:package.json express"
    """
    print(f"[TOOL] GitHub: Searching code with query '{query}'")
    
    params = {
        "q": query, 
        "per_page": min(max_results, 100)
    }
    
    try:
        result = make_github_request("GET", "/search/code", params=params)
        
        items = result.get("items", [])
        code_results = []
        
        for item in items:
            code_results.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "sha": item.get("sha"),
                "html_url": item.get("html_url"),
                "repository": item.get("repository", {}).get("full_name"),
                "repository_url": item.get("repository", {}).get("html_url")
            })
        
        return {
            "status": "success",
            "message": f"Found {result.get('total_count', 0)} code results (showing {len(code_results)})",
            "total_count": result.get("total_count", 0),
            "results": code_results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to search code: {str(e)}",
            "results": []
        }

import base64
from urllib.parse import quote

@tool
def github_update_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    commit_message: str,
    sha: str,
    branch: str = "main"
) -> dict:
    """
    Update an existing file in a GitHub repository
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: File path to update
        content: New content for the file
        commit_message: Commit message for this change
        sha: SHA hash of the file being replaced (get this from github_get_file_content)
        branch: Branch name (default: main)
    
    Returns:
        Dictionary with status and file information
    """
    print(f"[TOOL] GitHub: Updating file {path} in {owner}/{repo}")
    
    owner = clean_github_input(owner)
    repo = clean_github_input(repo)
    
    # Validate inputs
    if not sha:
        return {
            "status": "error",
            "message": "SHA is required for updating files. Use github_get_file_content to get the current SHA."
        }
    
    if not commit_message:
        return {
            "status": "error",
            "message": "Commit message is required"
        }
    
    # URL encode the path to handle special characters
    encoded_path = quote(path, safe='/')
    
    # Encode content to base64
    try:
        content_bytes = content.encode('utf-8')
        content_base64 = base64.b64encode(content_bytes).decode('utf-8')
        print(f"[DEBUG] Content encoded to base64 (length: {len(content_base64)})")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to encode content: {str(e)}"
        }
    
    # Prepare request body (SHA is required for updates)
    data = {
        "message": commit_message,
        "content": content_base64,
        "sha": sha,
        "branch": branch
    }
    
    print(f"[DEBUG] Update data prepared - SHA: {sha[:7]}..., Branch: {branch}")
    
    try:
        response = make_github_request(
            "PUT",
            f"/repos/{owner}/{repo}/contents/{encoded_path}",
            data=data
        )
        
        print(f"[DEBUG] File updated successfully")
        
        return {
            "status": "success",
            "message": f"File {path} updated successfully in {owner}/{repo}",
            "url": response.get("content", {}).get("html_url"),
            "new_sha": response.get("content", {}).get("sha"),
            "commit": {
                "sha": response.get("commit", {}).get("sha"),
                "url": response.get("commit", {}).get("html_url")
            }
        }
    except requests.exceptions.HTTPError as e:
        # Parse GitHub API error response
        error_msg = str(e)
        error_details = {}
        
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            error_details["status_code"] = status_code
            
            try:
                error_json = e.response.json()
                error_msg = error_json.get('message', str(e))
                error_details["api_message"] = error_msg
                
                # Check for specific error types
                if status_code == 404:
                    error_msg = f"Repository or file not found. Check: owner='{owner}', repo='{repo}', path='{path}', branch='{branch}'"
                elif status_code == 409:
                    error_msg = f"Conflict - SHA mismatch. The file may have been modified. Current SHA: {sha[:7]}..."
                elif status_code == 422:
                    error_msg = f"Invalid request. Check if SHA is correct and file exists: {error_json.get('message', '')}"
                elif status_code == 401 or status_code == 403:
                    error_msg = "Authentication failed or insufficient permissions. Check your GitHub token."
                    
            except:
                error_msg = e.response.text or str(e)
        
        print(f"[ERROR] Update failed: {error_msg}")
        
        return {
            "status": "error",
            "message": f"Failed to update file: {error_msg}",
            **error_details
        }
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


@tool
def github_delete_file(
    owner: str,
    repo: str,
    path: str,
    commit_message: str,
    sha: str,
    branch: str = "main"
) -> dict:
    """
    Delete a file from a GitHub repository
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: File path to delete
        commit_message: Commit message for this deletion
        sha: SHA hash of the file being deleted (get this from github_get_file_content)
        branch: Branch name (default: main)
    
    Returns:
        Dictionary with status information
    """
    print(f"[TOOL] GitHub: Deleting file {path} from {owner}/{repo}")
    
    owner = clean_github_input(owner)
    repo = clean_github_input(repo)
    
    # Validate inputs
    if not sha:
        return {
            "status": "error",
            "message": "SHA is required for deleting files. Use github_get_file_content to get the current SHA."
        }
    
    if not commit_message:
        return {
            "status": "error",
            "message": "Commit message is required"
        }
    
    # URL encode the path to handle special characters
    encoded_path = quote(path, safe='/')
    
    # Prepare request body
    data = {
        "message": commit_message,
        "sha": sha,
        "branch": branch
    }
    
    print(f"[DEBUG] Delete data prepared - SHA: {sha[:7]}..., Branch: {branch}")
    
    try:
        response = make_github_request(
            "DELETE",
            f"/repos/{owner}/{repo}/contents/{encoded_path}",
            data=data
        )
        
        print(f"[DEBUG] File deleted successfully")
        
        return {
            "status": "success",
            "message": f"File {path} deleted successfully from {owner}/{repo}",
            "commit": {
                "sha": response.get("commit", {}).get("sha"),
                "url": response.get("commit", {}).get("html_url")
            }
        }
    except requests.exceptions.HTTPError as e:
        # Parse GitHub API error response
        error_msg = str(e)
        error_details = {}
        
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            error_details["status_code"] = status_code
            
            try:
                error_json = e.response.json()
                error_msg = error_json.get('message', str(e))
                error_details["api_message"] = error_msg
                
                # Check for specific error types
                if status_code == 404:
                    error_msg = f"Repository or file not found. Check: owner='{owner}', repo='{repo}', path='{path}', branch='{branch}'"
                elif status_code == 409:
                    error_msg = f"Conflict - SHA mismatch. The file may have been modified. Current SHA: {sha[:7]}..."
                elif status_code == 422:
                    error_msg = f"Invalid request: {error_json.get('message', '')}"
                elif status_code == 401 or status_code == 403:
                    error_msg = "Authentication failed or insufficient permissions. Check your GitHub token."
                    
            except:
                error_msg = e.response.text or str(e)
        
        print(f"[ERROR] Delete failed: {error_msg}")
        
        return {
            "status": "error",
            "message": f"Failed to delete file: {error_msg}",
            **error_details
        }
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        
        }
@tool
def github_get_file_content(
    owner: str,
    repo: str,
    path: str,
    branch: str = "main",
    decode: bool = True
) -> dict:
    """
    Read/fetch the content of a file from a GitHub repository
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: File path in the repository
        branch: Branch name (default: main)
        decode: Whether to decode base64 content to text (default: True)
    
    Returns:
        Dictionary with file content, metadata, and SHA (needed for updates/deletes)
    """
    print(f"[TOOL] GitHub: Reading file {path} from {owner}/{repo}")
    
    owner = clean_github_input(owner)
    repo = clean_github_input(repo)
    
    # URL encode the path
    encoded_path = quote(path, safe='/')
    
    try:
        response = make_github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{encoded_path}?ref={branch}"
        )
        
        # Check if it's a file (not a directory)
        if response.get("type") != "file":
            return {
                "status": "error",
                "message": f"Path '{path}' is not a file, it's a {response.get('type')}"
            }
        
        # Get base64 encoded content
        content_base64 = response.get("content", "")
        sha = response.get("sha")
        
        print(f"[DEBUG] File found - SHA: {sha[:7]}..., Size: {response.get('size')} bytes")
        
        # Decode content if requested
        content = None
        if decode and content_base64:
            try:
                # Remove newlines from base64 (GitHub API adds them)
                content_base64_clean = content_base64.replace('\n', '')
                content = base64.b64decode(content_base64_clean).decode('utf-8')
            except UnicodeDecodeError:
                return {
                    "status": "warning",
                    "message": "File contains binary data, cannot decode as text",
                    "name": response.get("name"),
                    "path": response.get("path"),
                    "sha": sha,
                    "size": response.get("size"),
                    "url": response.get("html_url"),
                    "content": None,
                    "content_base64": content_base64
                }
        elif not decode:
            content = content_base64
        
        return {
            "status": "success",
            "message": f"File {path} read successfully",
            "name": response.get("name"),
            "path": response.get("path"),
            "sha": sha,  # IMPORTANT: This SHA is needed for updates and deletes
            "size": response.get("size"),
            "url": response.get("html_url"),
            "content": content,
            "encoding": response.get("encoding"),
            "download_url": response.get("download_url")
        }
        
    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        status_code = None
        
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            
            # Handle 404 specifically
            if status_code == 404:
                return {
                    "status": "error",
                    "message": f"File '{path}' not found in repository {owner}/{repo} on branch '{branch}'",
                    "status_code": 404
                }
            
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get('message', str(e))
            except:
                error_msg = e.response.text or str(e)
        
        print(f"[ERROR] Failed to read file: {error_msg}")
        
        return {
            "status": "error",
            "message": f"Failed to read file: {error_msg}",
            "status_code": status_code
        }
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }





DB_PATH = os.path.join(os.path.dirname(__file__), "expenses.db")

def init_db():
    """Initialize the expenses database."""
    with sqlite3.connect(DB_PATH) as c:
        c.execute('''CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT '')''')
        c.commit()

init_db()


@tool
def add_expense(
    date: str, 
    amount: float, 
    category: str, 
    subcategory: str = '', 
    note: str = ''
) -> dict:
    """Add a new expense to the tracker.
    
    Args:
        date: Date of expense in YYYY-MM-DD format
        amount: Amount spent (numeric value)
        category: Main expense category (e.g., food, transport, entertainment)
        subcategory: Optional subcategory for detailed tracking
        note: Optional note or description
    
    Returns:
        Status message confirming the expense was added
    """
    with sqlite3.connect(DB_PATH) as c:
        c.execute('''INSERT INTO expenses (date, amount, category, subcategory, note)
                     VALUES (?, ?, ?, ?, ?)''', (date, amount, category, subcategory, note))
        c.commit()
    return {"status": "success", "message": "Expense added successfully."}


@tool
def get_expenses(start_date: str, end_date: str) -> list:
    """Retrieve expenses within a date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        List of expense records matching the date range
    """
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            '''SELECT * FROM expenses WHERE date BETWEEN ? AND ?''', 
            (start_date, end_date)
        )
        expenses = cur.fetchall()
        cols = [description[0] for description in cur.description]
    return [dict(zip(cols, expense)) for expense in expenses]

@tool
def update_expense(
    expense_id: int,
    date: str = None,
    amount: float = None,
    category: str = None,
    subcategory: str = None,
    note: str = None
) -> dict:
    """Update an existing expense by its ID.
    
    Args:
        expense_id: The unique ID of the expense to update
        date: New date in YYYY-MM-DD format (optional)
        amount: New amount (optional)
        category: New category (optional)
        subcategory: New subcategory (optional)
        note: New note (optional)
    
    Returns:
        Status message confirming the update
    """
    with sqlite3.connect(DB_PATH) as c:
        # First, get the current expense to check if it exists
        cur = c.execute('SELECT * FROM expenses WHERE id = ?', (expense_id,))
        existing = cur.fetchone()
        
        if not existing:
            return {"status": "error", "message": f"Expense with ID {expense_id} not found."}
        
        # Build dynamic update query for only provided fields
        updates = []
        params = []
        
        if date is not None:
            updates.append("date = ?")
            params.append(date)
        if amount is not None:
            updates.append("amount = ?")
            params.append(amount)
        if category is not None:
            updates.append("category = ?")
            params.append(category)
        if subcategory is not None:
            updates.append("subcategory = ?")
            params.append(subcategory)
        if note is not None:
            updates.append("note = ?")
            params.append(note)
        
        if not updates:
            return {"status": "error", "message": "No fields provided to update."}
        
        params.append(expense_id)
        query = f"UPDATE expenses SET {', '.join(updates)} WHERE id = ?"
        
        c.execute(query, params)
        c.commit()
    
    return {"status": "success", "message": f"Expense ID {expense_id} updated successfully."}




@tool
def delete_expense(expense_id: int) -> dict:
    """Delete an expense by its ID.
    
    Args:
        expense_id: The unique ID of the expense to delete
    
    Returns:
        Status message confirming deletion
    """
    with sqlite3.connect(DB_PATH) as c:
        c.execute('''DELETE FROM expenses WHERE id = ?''', (expense_id,))
        c.commit()
    return {"status": "success", "message": "Expense deleted successfully."}


# List of all tools for easy import
expense_tools = [add_expense, get_expenses, delete_expense,update_expense]

from typing import Optional, Dict
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import re

class YouTubeToLinkedInTool:
    """Generate professional LinkedIn posts from YouTube videos"""
    
    def __init__(self, OPENROUTER_API_KEY: Optional[str] = None):
        """
        Initialize the tool with OpenRouter API key
        
        Args:
            OPENROUTER_API_KEY: OpenRouter API key
        """
        self.api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER API key required. Set OPENROUTER_API_KEY env variable or pass it directly")
        
        # Configure for OpenRouter
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            model="mistralai/mistral-7b-instruct:free",  # Free model
            temperature=0.7
        )
    
    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract YouTube video ID from various URL formats"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/shorts\/([^&\n?#]+)',
            r'youtu\.be\/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    @staticmethod
    def get_video_metadata(video_id: str) -> Dict[str, str]:
        """Get video title, description, and channel name using yt-dlp or pytube"""
        
        # Method 1: Try yt-dlp (more reliable)
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                return {
                    "title": info.get('title', 'Unknown Title'),
                    "description": info.get('description', ''),
                    "channel": info.get('uploader', info.get('channel', 'Unknown Channel')),
                    "length": info.get('duration', 0),
                    "views": info.get('view_count', 0),
                    "upload_date": info.get('upload_date', ''),
                }
        except ImportError:
            print(f"[INFO] yt-dlp not installed, trying pytube...")
        except Exception as e:
            print(f"[WARNING] yt-dlp failed: {str(e)}, trying pytube...")
        
        # Method 2: Fallback to pytube
        try:
            from pytube import YouTube
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return {
                "title": yt.title,
                "description": yt.description or "",
                "channel": yt.author,
                "length": yt.length,
                "views": yt.views
            }
        except ImportError:
            print(f"[WARNING] pytube not installed either")
        except Exception as e:
            print(f"[WARNING] pytube also failed: {str(e)}")
        
        # Method 3: Last resort - return video ID info
        return {
            "title": f"YouTube Video {video_id}",
            "description": f"Video ID: {video_id}. Install yt-dlp for better metadata: pip install yt-dlp",
            "channel": "Unknown Channel",
            "length": 0,
            "views": 0,
            "error": "Could not fetch metadata. Install yt-dlp: pip install yt-dlp"
        }
    
    @staticmethod
    def get_transcript(video_id: str, max_length: int = 5000) -> tuple[str, bool]:
        """
        Get video transcript/captions
        
        Args:
            video_id: YouTube video ID
            max_length: Maximum characters to return (for token limits)
        
        Returns:
            Tuple of (transcript_text, has_transcript)
        """
        try:
            # Try to get transcript in English
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # If English not available, get first available transcript
                available = list(transcript_list)
                if not available:
                    return ("", False)
                transcript = available[0]
            
            # Get the actual transcript data
            transcript_data = transcript.fetch()
            
            # Combine all text
            full_text = " ".join([entry['text'] for entry in transcript_data])
            
            # Truncate if too long
            if len(full_text) > max_length:
                full_text = full_text[:max_length] + "..."
            
            return (full_text, True)
            
        except Exception as e:
            print(f"[WARNING] Could not fetch transcript: {str(e)}")
            return ("", False)
    
    def generate_linkedin_post(
        self,
        video_url: str,
        post_style: str = "professional",
        include_hashtags: bool = True,
        max_length: int = 3000,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate LinkedIn post from YouTube video
        
        Args:
            video_url: YouTube video URL
            post_style: Style of post (professional, casual, thought-leader, storytelling)
            include_hashtags: Whether to include relevant hashtags
            max_length: Maximum post length (LinkedIn max is ~3000 chars)
            custom_prompt: Custom instructions for post generation
        
        Returns:
            Dictionary with post content and metadata
        """
        print(f"[TOOL] Processing YouTube video: {video_url}")
        
        try:
            # Extract video ID
            video_id = self.extract_video_id(video_url)
            print(f"[DEBUG] Video ID: {video_id}")
            
            # Get video metadata
            metadata = self.get_video_metadata(video_id)
            print(f"[DEBUG] Video Title: {metadata['title']}")
            
            # Get transcript (with fallback)
            transcript, has_transcript = self.get_transcript(video_id)
            
            # Check if we have ANY usable content
            has_description = metadata.get('description') and len(metadata.get('description', '').strip()) > 50
            has_title = metadata.get('title') and metadata.get('title') != "Unknown Title" and not metadata.get('title').startswith("YouTube Video")
            
            if has_transcript:
                print(f"[DEBUG] Transcript found - length: {len(transcript)} characters")
            elif has_description:
                print("[WARNING] No transcript, using video description")
                transcript = metadata.get('description', '')[:2000]
            elif has_title:
                print("[WARNING] No transcript or description, using title only")
                transcript = f"This video is titled: {metadata['title']}"
            else:
                return {
                    "status": "error",
                    "message": "Unable to fetch video content. The video may be private, age-restricted, or unavailable.",
                    "suggestion": "Try a different video or provide the video content manually.",
                    "video_url": video_url,
                    "video_id": video_id
                }
            
            # Prepare the prompt based on style
            style_instructions = {
                "professional": "Write in a professional, authoritative tone. Focus on insights and actionable takeaways.",
                "casual": "Write in a conversational, friendly tone. Make it relatable and engaging.",
                "thought-leader": "Write as a thought leader sharing deep insights. Be inspirational and forward-thinking.",
                "storytelling": "Use storytelling techniques. Start with a hook and build a narrative."
            }
            
            style_instruction = style_instructions.get(post_style, style_instructions["professional"])
            
            hashtags_instruction = "Include 3-5 relevant hashtags at the end" if include_hashtags else "Do not include hashtags"
            custom_instructions_text = custom_prompt if custom_prompt else "Follow standard LinkedIn best practices."
            
            # Adjust prompt based on whether we have transcript
            if has_transcript:
                content_source = f"VIDEO TRANSCRIPT:\n{transcript[:4000]}"
            else:
                content_source = f"VIDEO DESCRIPTION:\n{transcript}\n\nNote: Full transcript not available, create engaging post based on title and description."
            
            # Create the prompt
            prompt_text = f"""You are a professional LinkedIn content creator. Your task is to create an engaging LinkedIn post based on a YouTube video.

VIDEO DETAILS:
Title: {metadata['title']}
Channel: {metadata['channel']}

{content_source}

INSTRUCTIONS:
{style_instruction}

{custom_instructions_text}

Create a LinkedIn post that:
1. Captures the main insights and key takeaways from the video
2. Is engaging and provides value to LinkedIn audience
3. Has a strong opening hook to grab attention
4. Uses line breaks and emojis strategically for readability
5. Is maximum {max_length} characters
6. Includes a call-to-action at the end
7. Credits the original video creator
8. {hashtags_instruction}

Format the post to be ready to copy-paste directly into LinkedIn.

LINKEDIN POST:"""
            
            print("[DEBUG] Generating LinkedIn post with AI...")
            response = self.llm.invoke(prompt_text)
            post_content = response.content
            
            # Ensure length limit
            if len(post_content) > max_length:
                post_content = post_content[:max_length-3] + "..."
            
            return {
                "status": "success",
                "post": post_content,
                "video_title": metadata['title'],
                "video_url": video_url,
                "channel": metadata['channel'],
                "post_length": len(post_content),
                "style": post_style,
                "metadata": metadata
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] {error_details}")
            return {
                "status": "error",
                "message": f"Failed to generate LinkedIn post: {str(e)}",
                "video_url": video_url,
                "error_details": error_details
            }


# =============================================================================
# GLOBAL INITIALIZATION
# =============================================================================

# Initialize on import if API key is available
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
youtube_linkedin_generator = None

# Auto-initialize if API key exists
if OPENROUTER_API_KEY:
    try:
        youtube_linkedin_generator = YouTubeToLinkedInTool(OPENROUTER_API_KEY)
        print("[INFO] YouTube to LinkedIn tool initialized successfully")
    except Exception as e:
        print(f"[WARNING] Could not auto-initialize tool: {str(e)}")


# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

@tool
def generate_linkedin_post_from_youtube(
    video_url: str,
    post_style: str = "professional",
    include_hashtags: bool = True,
    custom_instructions: str = None
) -> str:
    """
    Generate a LinkedIn post from a YouTube video URL.
    
    Args:
        video_url: YouTube video URL (e.g., https://www.youtube.com/watch?v=... or https://youtu.be/...)
        post_style: Style of post - 'professional', 'casual', 'thought-leader', or 'storytelling'
        include_hashtags: Whether to include hashtags (True/False)
        custom_instructions: Additional instructions for post generation
    
    Returns:
        Generated LinkedIn post as string
    
    Example:
        generate_linkedin_post_from_youtube(
            "https://youtu.be/kCc8FmEb1nY",
            post_style="thought-leader",
            include_hashtags=True
        )
    """
    global youtube_linkedin_generator
    
    # Try to initialize if not already done
    if youtube_linkedin_generator is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            try:
                youtube_linkedin_generator = YouTubeToLinkedInTool(api_key)
                print("[INFO] Tool auto-initialized from environment variable")
            except Exception as e:
                return f"Error: Tool not initialized and auto-initialization failed: {str(e)}. Please set OPENROUTER_API_KEY environment variable or call initialize_tool(api_key) first."
        else:
            return "Error: Tool not initialized. Set OPENROUTER_API_KEY environment variable or call initialize_tool(api_key) first."
    
    result = youtube_linkedin_generator.generate_linkedin_post(
        video_url=video_url,
        post_style=post_style,
        include_hashtags=include_hashtags,
        custom_prompt=custom_instructions
    )
    
    if result["status"] == "success":
        return result["post"]
    else:
        return f"Error: {result['message']}\n\nDetails: {result.get('error_details', 'No additional details')}"


@tool
def get_youtube_video_summary(video_url: str) -> str:
    """
    Get a quick summary of a YouTube video's content.
    
    Args:
        video_url: YouTube video URL
    
    Returns:
        Summary of video content
    """
    global youtube_linkedin_generator
    
    # Try to initialize if not already done
    if youtube_linkedin_generator is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            try:
                youtube_linkedin_generator = YouTubeToLinkedInTool(api_key)
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return "Error: Tool not initialized. Set OPENROUTER_API_KEY environment variable."
    
    try:
        video_id = YouTubeToLinkedInTool.extract_video_id(video_url)
        metadata = YouTubeToLinkedInTool.get_video_metadata(video_id)
        transcript, has_transcript = YouTubeToLinkedInTool.get_transcript(video_id, max_length=3000)
        
        if not has_transcript:
            transcript = metadata.get('description', '')[:2000] or "No transcript or description available."
        
        # Generate quick summary
        prompt = f"""Provide a concise 3-4 sentence summary of this video:

Title: {metadata['title']}
Channel: {metadata['channel']}

Content:
{transcript[:2000]}

Summary:"""
        
        response = youtube_linkedin_generator.llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"
    
from typing import Annotated, Literal
import googlemaps
from datetime import datetime
import json
import os

# Initialize Google Maps client globally
GMAPS_CLIENT = None
INIT_ERROR = None  # Store initialization error

GOOGLE_MAPS_API_KEY = "AIzaSyBcHOsBtfXov8_7-w1rlrVT9mjCOMtNVnk"

print(f"DEBUG: API Key exists: {bool(GOOGLE_MAPS_API_KEY)}")
if GOOGLE_MAPS_API_KEY:
    print(f"DEBUG: API Key (first 10 chars): {GOOGLE_MAPS_API_KEY[:10]}...")
    try:
        GMAPS_CLIENT = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        print("✅ Google Maps client initialized successfully!")
    except Exception as e:
        INIT_ERROR = str(e)
        print(f"❌ [ERROR] Could not initialize client: {e}")
else:
    print("❌ [ERROR] GOOGLE_MAPS_API_KEY not found in environment!")

@tool
def get_traffic_info(
    origin: Annotated[str, "Starting location"],
    destination: Annotated[str, "Destination location"],
    mode: Annotated[Literal["driving", "walking", "bicycling", "transit"], "Travel mode"] = "driving"
) -> str:
    """Get real-time traffic information between two locations."""
    
    # Better error reporting
    if GMAPS_CLIENT is None:
        error_details = {
            "status": "error",
            "message": "Google Maps client not initialized",
            "api_key_set": bool(GOOGLE_MAPS_API_KEY),
            "initialization_error": INIT_ERROR,
            "troubleshooting": [
                "Check if GOOGLE_MAPS_API_KEY is set in environment",
                "Verify API key is valid",
                "Ensure Directions API is enabled in Google Cloud Console",
                "Confirm billing is set up (required even for free tier)"
            ]
        }
        return json.dumps(error_details, indent=2)
    
    try:
        now = datetime.now()
        
        print(f"DEBUG: Requesting directions from {origin} to {destination}")
        
        directions = GMAPS_CLIENT.directions(
            origin=origin,
            destination=destination,
            mode=mode,
            departure_time=now,
            alternatives=True,
            traffic_model="best_guess"
        )
        
        print(f"DEBUG: Got {len(directions) if directions else 0} routes")
        
        if not directions:
            return json.dumps({
                "status": "error",
                "message": f"No route found between {origin} and {destination}"
            })
        
        # Format response (same as before)
        routes_info = []
        
        for idx, route in enumerate(directions[:3]):
            leg = route['legs'][0]
            route_summary = route.get('summary', f'Route {idx + 1}')
            
            duration_normal = leg['duration']['text']
            duration_normal_sec = leg['duration']['value']
            
            duration_traffic_sec = leg.get('duration_in_traffic', {}).get('value', duration_normal_sec)
            duration_traffic = leg.get('duration_in_traffic', {}).get('text', duration_normal)
            
            distance = leg['distance']['text']
            delay_minutes = (duration_traffic_sec - duration_normal_sec) / 60
            
            if delay_minutes > 10:
                traffic_status = "heavy"
            elif delay_minutes > 5:
                traffic_status = "moderate"
            else:
                traffic_status = "light"
            
            route_info = {
                "route_name": route_summary,
                "distance": distance,
                "normal_duration": duration_normal,
                "current_duration": duration_traffic,
                "delay_minutes": int(delay_minutes),
                "traffic_status": traffic_status,
                "warnings": route.get('warnings', [])
            }
            
            routes_info.append(route_info)
        
        result = {
            "status": "success",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "routes": routes_info,
            "timestamp": now.isoformat()
        }
        
        return json.dumps(result, indent=2)
        
    except googlemaps.exceptions.ApiError as e:
        # Show the ACTUAL API error
        error_response = {
            "status": "error",
            "error_type": "Google Maps API Error",
            "message": str(e),
            "origin": origin,
            "destination": destination,
            "common_causes": [
                "Directions API not enabled in Google Cloud Console",
                "Invalid API key",
                "Billing not set up (required!)",
                "API key restrictions blocking the request",
                "Daily quota exceeded"
            ]
        }
        print(f"❌ API Error: {e}")  # Print to console for debugging
        return json.dumps(error_response, indent=2)
        
    except Exception as e:
        # Show ANY other error
        error_response = {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "origin": origin,
            "destination": destination
        }
        print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
        return json.dumps(error_response, indent=2)


from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional
import pytz

@tool
def get_current_datetime(timezone_str: str = "UTC", format: str = "full") -> str:
    """
    Get the current date and time.
    
    Args:
        timezone_str: Timezone (e.g., 'UTC', 'America/New_York', 'Asia/Kolkata', 'Europe/London')
        format: Output format - 'full', 'date', 'time', 'iso', 'unix'
    
    Returns:
        Current date/time in requested format
    
    Examples:
        get_current_datetime("America/New_York", "full")
        get_current_datetime("Asia/Kolkata", "date")
        get_current_datetime("UTC", "unix")
    """
    try:
        # Get timezone
        try:
            tz = ZoneInfo(timezone_str)
        except:
            # Fallback to pytz if zoneinfo doesn't work
            tz = pytz.timezone(timezone_str)
        
        now = datetime.now(tz)
        
        # Format based on request
        if format == "full":
            return now.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')
        elif format == "date":
            return now.strftime('%Y-%m-%d')
        elif format == "time":
            return now.strftime('%I:%M:%S %p')
        elif format == "iso":
            return now.isoformat()
        elif format == "unix":
            return str(int(now.timestamp()))
        else:
            return now.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')
            
    except Exception as e:
        return f"Error: {str(e)}. Try timezones like 'UTC', 'America/New_York', 'Asia/Kolkata'"


@tool
def get_date_info() -> str:
    """
    Get comprehensive information about the current date and time.

    Returns:
        Detailed date/time information including day of week, month, year, etc.
    """
    now = datetime.now()

    return f"""Current Date & Time Information:

📅 Date: {now.strftime('%A, %B %d, %Y')}
🕐 Time: {now.strftime('%I:%M:%S %p')}
🌍 Timezone: {now.strftime('%Z')}
📊 Week: Week {now.isocalendar()[1]} of {now.year}
📆 Day of Year: Day {now.timetuple().tm_yday} of 365/366
⏰ Unix Timestamp: {int(now.timestamp())}
🔗 ISO 8601: {now.isoformat()}
"""


# =============================================================================
# COLD EMAIL SYSTEM
# =============================================================================

# Database setup for cold emails
EMAIL_DB_PATH = os.path.join(os.path.dirname(__file__), "cold_emails.db")

def init_email_db():
    """Initialize the cold email database."""
    with sqlite3.connect(EMAIL_DB_PATH) as c:
        # Contacts table
        c.execute('''CREATE TABLE IF NOT EXISTS contacts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                company TEXT,
                position TEXT,
                linkedin_url TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # Email templates table
        c.execute('''CREATE TABLE IF NOT EXISTS email_templates(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                variables TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # Sent emails table
        c.execute('''CREATE TABLE IF NOT EXISTS sent_emails(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contact_id INTEGER,
                template_id INTEGER,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'sent',
                FOREIGN KEY (contact_id) REFERENCES contacts (id),
                FOREIGN KEY (template_id) REFERENCES email_templates (id))''')

        c.commit()

init_email_db()

def setup_gmail_auth():
    """Setup Gmail authentication for sending emails."""
    gmail_user = os.getenv("GMAIL_USER", "shraddha30405@gmail.com")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")  # App-specific password needed

    if not gmail_password:
        raise ValueError("GMAIL_APP_PASSWORD environment variable not set. "
                        "Please set up Gmail App Password and add it to your .env file.")

    return gmail_user, gmail_password

@tool
def add_contact(name: str, email: str, company: str = "", position: str = "", linkedin_url: str = "", notes: str = "") -> dict:
    """Add a new contact to the cold email database.

    Args:
        name: Contact's full name
        email: Contact's email address
        company: Company name
        position: Job position/title
        linkedin_url: LinkedIn profile URL
        notes: Additional notes about the contact

    Returns:
        Status message confirming contact addition
    """
    try:
        with sqlite3.connect(EMAIL_DB_PATH) as c:
            c.execute('''INSERT INTO contacts (name, email, company, position, linkedin_url, notes)
                         VALUES (?, ?, ?, ?, ?, ?)''', (name, email, company, position, linkedin_url, notes))
            c.commit()
        return {"status": "success", "message": f"Contact '{name}' added successfully."}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": f"Email '{email}' already exists in contacts."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to add contact: {str(e)}"}

@tool
def get_contacts(company_filter: str = "", limit: int = 50) -> list:
    """Retrieve contacts from the database.

    Args:
        company_filter: Filter by company name (optional)
        limit: Maximum number of contacts to return

    Returns:
        List of contact records
    """
    with sqlite3.connect(EMAIL_DB_PATH) as c:
        if company_filter:
            cur = c.execute(
                '''SELECT * FROM contacts WHERE company LIKE ? ORDER BY created_at DESC LIMIT ?''',
                (f'%{company_filter}%', limit)
            )
        else:
            cur = c.execute(
                '''SELECT * FROM contacts ORDER BY created_at DESC LIMIT ?''',
                (limit,)
            )
        contacts = cur.fetchall()
        cols = [description[0] for description in cur.description]
    return [dict(zip(cols, contact)) for contact in contacts]

@tool
def create_email_template(name: str, subject: str, body: str, variables: str = "") -> dict:
    """Create a new email template for cold outreach.

    Args:
        name: Template name
        subject: Email subject line
        body: Email body content (can include variables like {name}, {company}, etc.)
        variables: Comma-separated list of variables used in the template

    Returns:
        Status message confirming template creation
    """
    try:
        with sqlite3.connect(EMAIL_DB_PATH) as c:
            c.execute('''INSERT INTO email_templates (name, subject, body, variables)
                         VALUES (?, ?, ?, ?)''', (name, subject, body, variables))
            c.commit()
        return {"status": "success", "message": f"Email template '{name}' created successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to create template: {str(e)}"}

@tool
def get_email_templates() -> list:
    """Retrieve all email templates.

    Returns:
        List of email template records
    """
    with sqlite3.connect(EMAIL_DB_PATH) as c:
        cur = c.execute('''SELECT * FROM email_templates ORDER BY created_at DESC''')
        templates = cur.fetchall()
        cols = [description[0] for description in cur.description]
    return [dict(zip(cols, template)) for template in templates]

@tool
def send_cold_email(contact_email: str, template_name: str, custom_variables: dict = None) -> dict:
    """Send a cold email to a single contact using a template.

    Args:
        contact_email: Email address of the contact
        template_name: Name of the email template to use
        custom_variables: Dictionary of custom variables to replace in template

    Returns:
        Status message about email sending
    """
    try:
        # Setup Gmail authentication
        gmail_user, gmail_password = setup_gmail_auth()

        # Get contact information
        with sqlite3.connect(EMAIL_DB_PATH) as c:
            cur = c.execute('''SELECT * FROM contacts WHERE email = ?''', (contact_email,))
            contact = cur.fetchone()

        if not contact:
            return {"status": "error", "message": f"Contact with email '{contact_email}' not found."}

        contact_dict = dict(zip([desc[0] for desc in cur.description], contact))

        # Get email template
        with sqlite3.connect(EMAIL_DB_PATH) as c:
            cur = c.execute('''SELECT * FROM email_templates WHERE name = ?''', (template_name,))
            template = cur.fetchone()

        if not template:
            return {"status": "error", "message": f"Email template '{template_name}' not found."}

        template_dict = dict(zip([desc[0] for desc in cur.description], template))

        # Prepare email content
        subject = template_dict['subject']
        body = template_dict['body']

        # Replace variables in subject and body
        variables = {
            'name': contact_dict.get('name', ''),
            'company': contact_dict.get('company', ''),
            'position': contact_dict.get('position', ''),
            'email': contact_dict.get('email', ''),
            'linkedin_url': contact_dict.get('linkedin_url', ''),
        }

        if custom_variables:
            variables.update(custom_variables)

        for key, value in variables.items():
            subject = subject.replace(f'{{{key}}}', str(value))
            body = body.replace(f'{{{key}}}', str(value))

        # Create message
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = contact_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(gmail_user, gmail_password)
        text = msg.as_string()
        server.sendmail(gmail_user, contact_email, text)
        server.quit()

        # Record sent email
        with sqlite3.connect(EMAIL_DB_PATH) as c:
            c.execute('''INSERT INTO sent_emails (contact_id, template_id, subject, body)
                         VALUES (?, ?, ?, ?)''', (contact_dict['id'], template_dict['id'], subject, body))
            c.commit()

        return {
            "status": "success",
            "message": f"Email sent successfully to {contact_dict['name']} at {contact_email}",
            "recipient": contact_email,
            "subject": subject
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to send email: {str(e)}"}

@tool
def send_bulk_cold_emails(template_name: str, company_filter: str = "", max_emails: int = 10, delay_seconds: int = 5) -> dict:
    """Send bulk cold emails to multiple contacts.

    Args:
        template_name: Name of the email template to use
        company_filter: Filter contacts by company (optional)
        max_emails: Maximum number of emails to send
        delay_seconds: Delay between emails to avoid spam filters

    Returns:
        Status report of bulk email sending
    """
    import time

    try:
        # Get contacts
        contacts = get_contacts(company_filter=company_filter, limit=max_emails)

        if not contacts:
            return {"status": "error", "message": "No contacts found matching the criteria."}

        sent_count = 0
        failed_count = 0
        results = []

        for contact in contacts:
            result = send_cold_email(contact['email'], template_name)
            results.append({
                "email": contact['email'],
                "name": contact['name'],
                "status": result['status'],
                "message": result['message']
            })

            if result['status'] == 'success':
                sent_count += 1
            else:
                failed_count += 1

            # Delay between emails
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        return {
            "status": "completed",
            "message": f"Bulk email campaign completed. Sent: {sent_count}, Failed: {failed_count}",
            "total_contacts": len(contacts),
            "sent_count": sent_count,
            "failed_count": failed_count,
            "results": results
        }

    except Exception as e:
        return {"status": "error", "message": f"Bulk email campaign failed: {str(e)}"}

@tool
def get_sent_emails(limit: int = 50) -> list:
    """Retrieve sent email history.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of sent email records with contact and template information
    """
    with sqlite3.connect(EMAIL_DB_PATH) as c:
        cur = c.execute('''
            SELECT se.*, c.name as contact_name, c.company, et.name as template_name
            FROM sent_emails se
            JOIN contacts c ON se.contact_id = c.id
            JOIN email_templates et ON se.template_id = et.id
            ORDER BY se.sent_at DESC
            LIMIT ?
        ''', (limit,))
        emails = cur.fetchall()
        cols = [description[0] for description in cur.description]
    return [dict(zip(cols, email)) for email in emails]

# Cold email tools list
cold_email_tools = [add_contact, get_contacts, create_email_template, get_email_templates,
                   send_cold_email, send_bulk_cold_emails, get_sent_emails]


# Import Cloud Image Generation (No local ComfyUI required!)
from cloud_image_generation import generate_marketing_image_cloud

@tool
def generate_marketing_image(
    prompt: str,
    style: str = "marketing_poster",
    aspect_ratio: str = "1:1",
    provider: str = "auto"
):
    """
    Generate marketing images using cloud AI services for startups and small businesses.
    No local ComfyUI setup required!

    Args:
        prompt: Description of the image you want to generate
        style: Style of image - 'marketing_poster' or 'social_media'
        aspect_ratio: Aspect ratio - '1:1' (square) or '9:16' (portrait)
        provider: AI service - 'auto', 'openai', 'stability', 'replicate', 'huggingface'

    Returns:
        Dictionary with image data and metadata for frontend display
    """
    print(f"[DEBUG] generate_marketing_image called with prompt: {prompt}")

    try:
        # Actually call the cloud generation service
        result = generate_marketing_image_cloud(prompt, style, aspect_ratio, provider)

        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"🎨 Image generated successfully using {result.get('provider', 'cloud service')}!",
                "prompt": prompt,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "provider": result.get('provider', 'Unknown'),
                "model": result.get('model', 'Unknown'),
                "image_base64": result.get('image_base64'),
                "image_data": result.get('image_data'),
                "filename": f"generated_{prompt[:30].replace(' ', '_')}.png"
            }
        else:
            return {
                "status": "error",
                "message": result['message'],
                "error_type": "generation_failed",
                "suggestion": "Check that you have API keys configured in your .env file. Available providers: OpenAI (OPENAI_API_KEY), Stability AI (STABILITY_API_KEY), Replicate (REPLICATE_API_TOKEN), or Hugging Face (HUGGINGFACE_API_KEY - optional)."
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error during image generation: {str(e)}",
            "error_type": "unexpected_error",
            "suggestion": "Make sure you have at least one cloud image generation service configured and your API keys are valid."
        }

tools=[search_tool,calculator,get_stock_price,get_weather,rag_tool,github_create_repository,github_get_file_content,
       github_delete_repository, github_search_code,github_update_repository,github_get_repository,
       github_search_repositories,github_list_user_repositories,add_expense,
       get_expenses, delete_expense,update_expense,github_create_file,github_delete_file,github_update_file
       ,generate_linkedin_post_from_youtube, get_youtube_video_summary,get_traffic_info,get_current_datetime,get_date_info,
       add_contact, get_contacts, create_email_template, get_email_templates,
       send_cold_email, send_bulk_cold_emails, get_sent_emails, generate_marketing_image]

tool_node=ToolNode(tools)

load_dotenv()


model = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",)

llm_with_tool=model.bind_tools(tools)


conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)


class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:chat_state,config=None):
     """LLM node that may answer or request a tool call."""
     thread_id = None
     if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

     system_message = SystemMessage(
        content=(
            "You are a helpful assistant with cold email capabilities. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF. An expense tracking tool is also provided to keep track of personal expenses. "
            "When user asks to delete an expense, do not ask for the id - find it by calling get_expenses tool according to the information given by user. "
            "You can send cold emails using the Gmail integration - use add_contact to add HR contacts, create_email_template to create templates, "
            "and send_cold_email or send_bulk_cold_emails to send personalized cold outreach emails to HR representatives."
        )
    )


     messages=[system_message,*state['messages']]
     response=llm_with_tool.invoke(messages,config=config)
     return {'messages':[response]}

graph=StateGraph(chat_state)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')



chatbot=graph.compile(checkpointer=checkpointer)


 ## print(chatbot.invoke({'messages':[HumanMessage(content='hi my name is himanshu')]}, {'configurable':{'thread_id':"2"}}))
a=set()
for checkpoint in checkpointer.list(None):
    a.add(checkpoint.config['configurable']['thread_id'])
b=list(a)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


