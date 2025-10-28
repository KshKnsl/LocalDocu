import requests
from flask import Flask, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

OLLAMA_URL = "https://mari-unbequeathed-milkily.ngrok-free.app"

@app.route("/api/<path:path>", methods=["GET", "POST", "OPTIONS"])
def proxy(path):
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin", "*")
        cors_headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600",
            "Vary": "Origin",
        }
        return Response("", status=204, headers=cors_headers)

    url = f"{OLLAMA_URL}/{path}"
    headers = {"User-Agent": "Mozilla/5.0"}
    headers.update({k: v for k, v in request.headers.items() if k in ["Authorization", "Content-Type"]})

    if request.method == "POST":
        data = request.get_json() if request.is_json else request.get_data()
        resp = requests.post(url, json=data if request.is_json else None, data=None if request.is_json else data, headers=headers)
    else:
        resp = requests.get(url, params=request.args, headers=headers)

    # Merge upstream headers and ensure CORS headers are present.
    origin = request.headers.get("Origin")
    response_headers = {**resp.headers}
    if origin:
        response_headers["Access-Control-Allow-Origin"] = origin
        response_headers["Vary"] = "Origin"
    else:
        response_headers.setdefault("Access-Control-Allow-Origin", "*")

    response_headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    response_headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response_headers.setdefault("Access-Control-Allow-Credentials", "true")
    response_headers.setdefault("Access-Control-Max-Age", "3600")
    return Response(resp.content, status=resp.status_code, headers=response_headers)

@app.route('/')
def home():
    return 'Hello, World!'


# Ensure CORS headers are set on every response as a final fallback.
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    else:
        response.headers.setdefault("Access-Control-Allow-Origin", "*")

    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.setdefault("Access-Control-Allow-Credentials", "true")
    response.headers.setdefault("Access-Control-Max-Age", "3600")
    return response