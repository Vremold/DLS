import os
import sys
sys.path.append("..")
import json

from flask import Flask
from flask import request
from System import UI
from pathutil import PROJECT_ABS_DIR

# system = UI()
app = Flask(__name__)

"""Sample response for testing API"""
sample_payloads = [{
    "repo_name": "theon", 
    "repo_owner": "theonjs",
    "repo_url": "https://github.com/theonjs/theon",
    "repo_des": "Declarative library to build Web API clients & SDKs for the browser and node.js",
    "topics": ["http-agent", "http-client", "middleware", "javascript", "nodejs", "extensible", "declarative"]
}] * 20

""""""
GitUrl2Info = dict()
with open("../Data/GH/CleanedData/repositories.txt", "r", encoding="utf-8") as inf:
    for line in inf:
        line = json.loads(line)
        GitUrl2Info[line["git_url"]] = (line["git_topics"], line["git_description"])

def get_payloads(repos):
    payloads = []
    for r in repos:
        try:
            repo_owner, repo_name = r.strip().split("#")
            repo_url = f"https://github.com/{repo_owner}/{repo_name}"
            (repo_topics, repo_des) = GitUrl2Info[repo_url]
            payloads.append({
                "repo_name": repo_name, 
                "repo_owner": repo_owner,
                "repo_url": repo_url,
                "repo_des": repo_des,
                "topics": repo_topics
            })
        except Exception as e:
            return None
    return payloads
        

@app.route("/")
def index():
    return "Thanks for using Semantical Library Retrieval"

@app.route("/favicon.ico")
def favicon():
    return ""

@app.route("/search_demo", methods=["GET", "POST"])
def smaple_search():
    if request.method == "GET":
        text = request.args.get("query", "")
    elif request.method == "POST":
        text = request.form.get("query", "")
    print(text)
    if text == "":
        return {
            "success": False,
            "payload": None,
            "message": "Invalid Query!"
        }
    return {
        "success": True,
        "payload": sample_payloads,
        "message": "Retrieval succeed!"
    }

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "GET":
        text = request.args.get("query", "")
    elif request.method == "POST":
        text = request.form.get("query", "")
    print(text)
    if text == "":
        return {
            "success": False,
            "payload": None,
            "message": "Invalid Query!"
        }
    # text = ""
    # repos = system.query_for_ui(text)
    repos = ["theonjs#theon"] * 20
    
    payloads = get_payloads(repos)
    if not payloads:
        return {
            "success": False,
            "payload": None,
            "message": "Failed Retrieval!"
        }
    return {
        "success": True,
        "payload": payloads,
        "message": "Retrieval Succeed!"
    }

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)