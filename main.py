from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import jwt
import json
import hashlib
import time
from datetime import datetime, timedelta
import httpx
import logging
import os
from typing import Optional, Dict
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import jwt
import json
import hashlib
import time
from datetime import datetime, timedelta
import httpx
import logging
import os
import asyncio
import base64
import mimetypes
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
import google.generativeai as genai
import anthropic
import numpy as np
from PIL import Image
import io
import av
from pathlib import Path


# Initialize
app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
security = HTTPBearer()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Enhanced Model Configurations
# Config
SECRET_KEY = "your-secret-key-change-in-production"
API_KEYS = {
    "OPENAI": os.environ.get("opene", "default-key"),  # Added default value
    "ANTHROPIC": os.environ.get("secretant", "default-key")  # Added default value
}
COSTS = {
    "gpt4o": {"input": 0.0025, "output": 0.00125},
    "gpt4o-mini": {"input": 0.00015, "output": 0.000075},
    "claude": {"input": 0.003, "output": 0.00375}
}

class LoginData(BaseModel):
    username: str
    password: str

class SolveRequest(BaseModel):
    text: str
    model: str = "gpt4o-mini"

class UpdateCreditsRequest(BaseModel):
    user_id: str
    amount: float

class CreateUserRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False

class ResetPasswordRequest(BaseModel):
    password: str

# Setup DB
os.makedirs("db", exist_ok=True)
USERS_FILE = "db/users.json"
USAGE_FILE = "db/usage.json"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({"items": [
            {"id": "admin", "username": "admin", "password_hash": hashlib.sha256("admin123".encode()).hexdigest(), 
             "credits": 1000.0, "is_admin": True},
            {"id": "user", "username": "user", "password_hash": hashlib.sha256("user123".encode()).hexdigest(), 
             "credits": 100.0, "is_admin": False}
        ]}, f)

if not os.path.exists(USAGE_FILE):
    with open(USAGE_FILE, 'w') as f:
        json.dump({"items": []}, f)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Helper functions
def get_user(username: str) -> Optional[Dict]:
    with open(USERS_FILE) as f:
        return next((u for u in json.load(f)["items"] if u["username"] == username), None)

def update_user(user: Dict):
    with open(USERS_FILE) as f:
        data = json.load(f)
    idx = next(i for i, u in enumerate(data["items"]) if u["id"] == user["id"])
    data["items"][idx] = user
    with open(USERS_FILE, 'w') as f:
        json.dump(data, f)

def calculate_cost(model: str, chars: int) -> float:
    cost = COSTS[model]
    return ((chars / 1000) * cost["input"] + (chars * 1.5 / 1000) * cost["output"]) * 40

# Routes
@app.get("/admin")
async def admin_ui():
    with open("templates/admin.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/login")
async def login(data: LoginData):
    user = get_user(data.username)
    if not user or user["password_hash"] != hashlib.sha256(data.password.encode()).hexdigest():
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "token": jwt.encode({"user_id": user["id"], "exp": datetime.utcnow() + timedelta(days=1)}, SECRET_KEY),
        "user": {k: v for k, v in user.items() if k != "password_hash"}
    }


def compress_js(js_content):
    """
    Compress JavaScript by:
    1. Removing comments
    2. Removing unnecessary whitespace
    3. Minifying the code
    """
    # Remove single-line comments
    js_content = re.sub(r'//.*', '', js_content)

    # Remove multi-line comments
    js_content = re.sub(r'/\*.*?\*/', '', js_content, flags=re.DOTALL)

    # Remove leading/trailing whitespace on each line
    js_content = '\n'.join(line.strip() for line in js_content.split('\n'))

    # Remove unnecessary whitespace between tokens
    js_content = re.sub(r'\s+', ' ', js_content)

    # Remove unnecessary spaces around operators and brackets
    js_content = re.sub(r'\s*([(){}\[\]=+\-*/])\s*', r'\1', js_content)

    # Remove newlines
    js_content = js_content.replace('\n', ' ')

    return js_content.strip()

@app.get("/exte", response_class=HTMLResponse)
async def question_solver_page():
    """
    Serves the Question Solver bookmarklet installation page
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Question Solver Bookmarklet</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            line-height: 1.6; 
        }
        h1 { color: #333; }
        .bookmarklet-button {
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: move;
        }
        .instructions {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Question Solver Bookmarklet</h1>
    <div class="instructions">
        <h2>Installation Steps:</h2>
        <h3>1. Show Bookmarks Bar</h3>
        <p>If your bookmarks bar is hidden:</p>
        <ul>
            <li>Chrome/Edge: Press Ctrl+Shift+B (Windows/Linux) or Cmd+Shift+B (Mac)</li>
            <li>Firefox: Press Ctrl+Shift+B (Windows/Linux) or Cmd+Shift+B (Mac)</li>
        </ul>
        <h3>2. Drag to Bookmarks Bar</h3>
        <p>Click and drag the button below to your bookmarks bar:</p>
        <a href="javascript:(function(){const s=document.createElement('script');s.src='https://nicee.up.railway.app/script.js';document.body.appendChild(s);})();" class="bookmarklet-button">Question Solver</a>
        <h3>3. Use the Bookmarklet</h3>
        <p>Click the "Question Solver" bookmark on any webpage to activate</p>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/script.js")
async def get_script():
    return FileResponse("script.js", media_type="application/javascript")



@app.post("/api/admin/users/{user_id}/reset_password")
async def reset_password(user_id: str, request: ResetPasswordRequest, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        admin = get_user(payload["user_id"])
        if not admin or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        with open(USERS_FILE, 'r+') as f:
            data = json.load(f)
            user = next((u for u in data["items"] if u["id"] == user_id), None)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            user["password_hash"] = hashlib.sha256(request.password.encode()).hexdigest()
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

        return {"message": "Password reset successfully"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: str, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        admin = get_user(payload["user_id"])
        if not admin or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        with open(USERS_FILE, 'r+') as f:
            data = json.load(f)
            data["items"] = [user for user in data["items"] if user["id"] != user_id]
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

        return {"message": "User deleted successfully"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/admin/users")
async def create_user(request: CreateUserRequest, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        admin = get_user(payload["user_id"])
        if not admin or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        new_user = {
            "id": request.username,
            "username": request.username,
            "password_hash": hashlib.sha256(request.password.encode()).hexdigest(),
            "credits": 0.0,
            "is_admin": request.is_admin
        }

        with open(USERS_FILE, 'r+') as f:
            data = json.load(f)
            data["items"].append(new_user)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

        return {"message": "User created successfully"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/api/solve")
async def solve_question(request: SolveRequest, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Verify token and get user
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        # Calculate cost
        cost = calculate_cost(request.model, len(request.text))

        # Check credits if not free tier
        if not user.get("is_free_tier") and user["credits"] < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        # Make API request based on model
        if request.model.startswith("gpt"):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEYS['OPENAI']}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4-turbo-preview" if request.model == "gpt4o" else "gpt-3.5-turbo",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert at answering questions. Provide your response as a valid JSON object with exactly these three fields: 'answer' (a concise answer), 'explanation' (a brief explanation), and 'confidence' (a number 0-100). Example: {\"answer\": \"ampere\", \"explanation\": \"Current is measured in amperes (A)\", \"confidence\": 95}"
                            },
                            {
                                "role": "user",
                                "content": request.text
                            }
                        ]
                    }
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="OpenAI API error")

                response_content = response.json()["choices"][0]["message"]["content"]
                try:
                    # Handle both string and dict responses
                    if isinstance(response_content, dict):
                        result = response_content
                    else:
                        # Remove any potential leading/trailing whitespace
                        response_content = response_content.strip()
                        # Parse the string response
                        result = json.loads(response_content)

                    # Ensure required fields exist
                    if not all(key in result for key in ['answer', 'explanation', 'confidence']):
                        raise ValueError("Missing required fields in response")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse GPT response: {response_content}")
                    result = {
                        "answer": "Error parsing response",
                        "explanation": "The model returned an invalid format",
                        "confidence": 0
                    }

        else:  # Claude
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": API_KEYS["ANTHROPIC"],
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "claude-3-sonnet-20240229",
                        "max_tokens": 150,
                        "messages": [{
                            "role": "user",
                            "content": f"Answer this question and format the response exactly like this example, nothing else: {{'answer': 'ampere', 'explanation': 'Current is measured in amperes (A)', 'confidence': 95}}\n\nQuestion: {request.text}"
                        }]
                    }
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="Anthropic API error")

                response_content = response.json()["content"][0]["text"]
                try:
                    # Handle both string and dict responses
                    if isinstance(response_content, dict):
                        result = response_content
                    else:
                        # Remove any potential leading/trailing whitespace
                        response_content = response_content.strip()
                        # Parse the string response
                        result = json.loads(response_content)

                    # Ensure required fields exist
                    if not all(key in result for key in ['answer', 'explanation', 'confidence']):
                        raise ValueError("Missing required fields in response")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Claude response: {response_content}")
                    result = {
                        "answer": "Error parsing response",
                        "explanation": "The model returned an invalid format",
                        "confidence": 0
                    }

        # Deduct credits if not free tier
        if not user.get("is_free_tier"):
            user["credits"] -= cost
            update_user(user)

        # Log usage
        with open(USAGE_FILE) as f:
            usage = json.load(f)
        usage["items"].append({
            "user_id": user["id"],
            "timestamp": datetime.utcnow().isoformat(),
            "model": request.model,
            "cost": cost,
            "chars": len(request.text)
        })
        with open(USAGE_FILE, "w") as f:
            json.dump(usage, f)

        return {
            "response": result,
            "cost": cost,
            "cached": False
        }

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Error in solve_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/users/{user_id}/usage")
async def get_user_usage(user_id: str, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        admin = get_user(payload["user_id"])
        if not admin or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        with open(USAGE_FILE) as f:
            usage_data = json.load(f)
            user_usage = [item for item in usage_data["items"] if item["user_id"] == user_id]

        return user_usage
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/admin/users")
async def get_users(auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user or not user["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        with open(USERS_FILE) as f:
            return json.load(f)["items"]
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/admin/usage")
async def get_usage(auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user or not user["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        with open(USAGE_FILE) as f:
            return json.load(f)["items"]
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/admin/update_credits")
async def update_user_credits(request: UpdateCreditsRequest, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        admin = get_user(payload["user_id"])
        if not admin or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        user = get_user(request.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user["credits"] += request.amount
        update_user(user)
        return {"message": "Credits updated successfully"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Add verification endpoint
@app.get("/api/verify")
async def verify_token(auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        return {"valid": True}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)  # Changed port to 
