import requests
from flask import Flask, request, Response

app = Flask(__name__)

OLLAMA_URL = "https://mari-unbequeathed-milkily.ngrok-free.app"

@app.route("/api/<path:path>", methods=["GET", "POST", "OPTIONS"])
def proxy(path):
    if request.method == "OPTIONS":
        return Response(
            "",
            status=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
            },
        )


    url = f"{OLLAMA_URL}/{path}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Do NOT forward Origin or Referer (these can cause 403 with ngrok/Ollama)
    if "Authorization" in request.headers:
        headers["Authorization"] = request.headers["Authorization"]
    if "Content-Type" in request.headers:
        headers["Content-Type"] = request.headers["Content-Type"]

    if request.method == "POST":
        if request.is_json:
            resp = requests.post(url, json=request.get_json(), headers=headers, stream=True)
        else:
            resp = requests.post(url, data=request.get_data(), headers=headers, stream=True)
    else:
        resp = requests.get(url, params=request.args, headers=headers, stream=True)

    response_headers = dict(resp.headers)
    response_headers["Access-Control-Allow-Origin"] = "*"
    response_headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response_headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"

    return Response(resp.content, status=resp.status_code, headers=response_headers)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'