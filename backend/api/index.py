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
    headers = {"User-Agent": "Mozilla/5.0"}
    headers.update({k: v for k, v in request.headers.items() if k in ["Authorization", "Content-Type"]})

    if request.method == "POST":
        data = request.get_json() if request.is_json else request.get_data()
        resp = requests.post(url, json=data if request.is_json else None, data=None if request.is_json else data, headers=headers)
    else:
        resp = requests.get(url, params=request.args, headers=headers)

    response_headers = {**resp.headers,
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization"
    }
    return Response(resp.content, status=resp.status_code, headers=response_headers)

@app.route('/')
def home():
    return 'Hello, World!'