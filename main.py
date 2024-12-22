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
import base64
from typing import Optional, List
import os
from datetime import datetime
import uuid
from fastapi.responses import StreamingResponse
import asyncio

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
MODELS = {
    'claude-3-opus-latest': {
        'api_name': 'claude-3-opus-20240229',
        'cost': {'input': 0.015, 'output': 0.075},
        'supports_vision': True
    },
    'claude-3-sonnet-latest': {
        'api_name': 'claude-3-5-sonnet-20241022',
        'cost': {'input': 0.003, 'output': 0.015},
        'supports_vision': True
    },
    'claude-3-haiku-latest': {
        'api_name': 'claude-3-5-haiku-20241022',
        'cost': {'input': 0.0015, 'output': 0.0075},
        'supports_vision': True
    },
    'gpt4o': {
        'api_name': 'gpt-4o',
        'cost': {'input': 0.01, 'output': 0.03},
        'supports_vision': True
    },
    'gpt4o-mini': {
        'api_name': 'gpt-4o-mini',
        'cost': {'input': 0.005, 'output': 0.015},
        'supports_vision': True
    },
    'o1': {
        'api_name': 'o1',
        'cost': {'input': 0.02, 'output': 0.06},
        'supports_vision': False
    },
    'o1-mini': {
        'api_name': 'o1-mini',
        'cost': {'input': 0.01, 'output': 0.03},
        'supports_vision': False
    }
}
# Enhanced Request/Response Models
class LoginData(BaseModel):
    username: str
    password: str

class SolveRequest(BaseModel):
    text: str
    model: str = "gpt4o-mini"
    context: Optional[Dict] = None
    imageData: Optional[Dict] = None

class ChatRequest(BaseModel):
    message: str
    model: str
    history: List[Dict] = []

class ImageAnalysisRequest(BaseModel):
    image: str
    question: Optional[str] = None

class ContextRequest(BaseModel):
    elements: List[Dict]
    question: str

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


# Add these global variables
CHATS_FILE = "db/chats.json"
UPLOADS_DIR = "uploads"
os.makedirs("db", exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

if not os.path.exists(CHATS_FILE):
    with open(CHATS_FILE, 'w') as f:
        json.dump({"chats": []}, f)




async def stream_o1(messages: List[dict], model: str):
    """Stream responses from O1 reasoning models using OpenAI API"""
    try:
        async with httpx.AsyncClient() as client:
            # Convert messages to OpenAI format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg["content"], list):
                    # Handle multimodal content
                    content = []
                    for item in msg["content"]:
                        if item["type"] == "text":
                            content.append(item["text"])
                        elif item["type"] == "image":
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{item['source']['data']}"
                                }
                            })
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": content[0] if len(content) == 1 else content
                    })
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('opene')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-turbo-preview",  # Base model for O1
                    "messages": [
                        {"role": "system", "content": "You are an advanced reasoning assistant focused on step-by-step logical analysis and problem-solving. Take your time to think through problems carefully."}, 
                        *formatted_messages
                    ],
                    "stream": True,
                    "temperature": 0.1,  # Lower temperature for reasoning
                    "max_tokens": 4096
                },
                timeout=None
            )

            if response.status_code != 200:
                logger.error(f"O1 API error: {response.status_code}")
                raise HTTPException(status_code=response.status_code)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "content" in data:
                            yield {"content": data["content"]}
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        logger.error(f"Error in stream_o1: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# New route to serve the AI chat interface
@app.get("/ai")
async def serve_chat_ui():
    with open("templates/ai.html", "r") as f:
        return HTMLResponse(content=f.read())

# Chat management functions
def get_user_chats(user_id: str) -> List[dict]:
    with open(CHATS_FILE, 'r') as f:
        data = json.load(f)
        return [chat for chat in data["chats"] if chat["user_id"] == user_id]

def save_chat(chat: dict):
    with open(CHATS_FILE, 'r+') as f:
        data = json.load(f)
        chat_idx = next((i for i, c in enumerate(data["chats"]) 
                        if c["id"] == chat["id"]), None)
        if chat_idx is not None:
            data["chats"][chat_idx] = chat
        else:
            data["chats"].append(chat)
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)

@app.post("/api/chats")
async def create_chat(auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        
        chat = {
            "id": str(uuid.uuid4()),
            "user_id": user["id"],
            "title": "New Chat",
            "created_at": datetime.utcnow().isoformat(),
            "messages": []
        }
        save_chat(chat)
        return chat
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/chats")
async def get_chats(auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        return get_user_chats(user["id"])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.put("/api/chats/{chat_id}/title")
async def update_chat_title(
    chat_id: str, 
    title: str,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
            
        chat = next((c for c in get_user_chats(user["id"]) 
                    if c["id"] == chat_id), None)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
            
        chat["title"] = title
        save_chat(chat)
        return {"status": "success"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        
        with open(CHATS_FILE, 'r+') as f:
            data = json.load(f)
            data["chats"] = [c for c in data["chats"] 
                           if not (c["id"] == chat_id and c["user_id"] == user["id"])]
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
        return {"status": "success"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/api/upload")
async def upload_file(
    files: List[UploadFile],
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        
        uploaded_files = []
        for file in files:
            # Increased max size to 20MB
            if len(await file.read()) > 20 * 1024 * 1024:  
                raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 20MB limit")
            await file.seek(0)
            
            content = await file.read()
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            # Expanded supported formats
            if file.content_type.startswith('image/') or file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                # Process image
                try:
                    img = Image.open(io.BytesIO(content))
                    # Resize if needed while maintaining aspect ratio
                    if max(img.size) > 1568:
                        ratio = 1568 / max(img.size)
                        new_size = tuple(int(dim * ratio) for dim in img.size)
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Convert to RGB if needed
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    # Save as JPEG
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG', quality=85)
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
                    uploaded_files.append({
                        "type": "image",
                        "name": file.filename,
                        "base64": f"data:image/jpeg;base64,{base64_image}"
                    })
                except Exception as e:
                    logger.error(f"Image processing error: {str(e)}")
                    raise HTTPException(status_code=400, detail="Invalid image file")
            else:
                # Handle other file types
                uploaded_files.append({
                    "type": "file",
                    "name": file.filename,
                    "content": base64.b64encode(content).decode('utf-8')
                })

        return uploaded_files
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_gpt(messages: List[dict], model: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('opene')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o" if model == "gpt4o" else "gpt-4o-mini",
                "messages": messages,
                "stream": True
            },
            timeout=None
        )
        
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                if line.strip() == 'data: [DONE]':
                    break
                try:
                    chunk = json.loads(line[6:])
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                except json.JSONDecodeError:
                    continue

async def stream_claude(messages: List[dict], model: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.getenv('secretant'),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": CLAUDE_MODELS[model],
                    "messages": messages,
                    "max_tokens": 4096,
                    "stream": True
                },
                timeout=None
            )

            if response.status_code != 200:
                logger.error(f"Claude API error: {response.status_code}")
                logger.error(f"Response: {await response.text()}")
                raise HTTPException(status_code=response.status_code)

            buffer = ""
            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        
                        # Handle different event types
                        if data["type"] == "message_start":
                            continue
                        elif data["type"] == "content_block_start":
                            continue
                        elif data["type"] == "content_block_delta":
                            if "delta" in data and data["delta"]["type"] == "text_delta":
                                text = data["delta"]["text"]
                                if text:
                                    yield f"data: {json.dumps({'content': text})}\n\n"
                        elif data["type"] == "content_block_stop":
                            continue
                        elif data["type"] == "message_delta":
                            continue
                        elif data["type"] == "message_stop":
                            break
                        elif data["type"] == "error":
                            logger.error(f"Claude streaming error: {data}")
                            raise HTTPException(status_code=500, detail=data["error"])
                        elif data["type"] == "ping":
                            continue

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}, Line: {line}")
                        continue

    except Exception as e:
        logger.error(f"Error in stream_claude: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


CLAUDE_MODELS = {
    "claude-opus": "claude-3-opus-20240229",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "claude-haiku": "claude-3-5-haiku-20241022"
}


async def process_file_costs(files):
    total_chars = 0
    for file in files:
        if file["type"] == "image":
            # Base cost for each image (similar to Vision API)
            total_chars += 1000  # Base char count for images
    return total_chars


@app.post("/api/chat/stream")
async def stream_chat(request: dict, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        model = request.get('model')
        if model not in MODELS:
            raise HTTPException(status_code=400, detail="Invalid model")

        # Calculate initial cost based on input
        message_chars = len(request["message"])
        file_chars = await process_file_costs(request.get("files", []))
        total_input_chars = message_chars + file_chars
        
        # Estimate initial cost (will be updated with actual output length)
        estimated_cost = calculate_cost(model, total_input_chars)

        if not user.get("is_free_tier") and user["credits"] < estimated_cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        # Format messages for model API
        messages = []
        for msg in request.get("history", []):
            content = []
            if isinstance(msg.get("content"), dict):
                if "files" in msg["content"]:
                    for file in msg["content"]["files"]:
                        if file["type"] == "image" and MODELS[model]['supports_vision']:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": file["base64"].split(",")[1]
                                }
                            })
                content.append({"type": "text", "text": msg["content"].get("text", "")})
            else:
                content.append({"type": "text", "text": msg["content"]})
            
            messages.append({
                "role": msg["role"],
                "content": content
            })

        # Add current message with files
        current_content = []
        if request.get("files"):
            for file in request.get("files", []):
                if file["type"] == "image" and MODELS[model]['supports_vision']:
                    current_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": file["base64"].split(",")[1]
                        }
                    })
        current_content.append({"type": "text", "text": request["message"]})
        messages.append({
            "role": "user",
            "content": current_content
        })

        # Stream based on model type
        if model.startswith('claude'):
            stream = stream_claude(messages, MODELS[model]['api_name'])
        elif model.startswith('gpt'):
            stream = stream_gpt(messages, MODELS[model]['api_name'])
        else:  # o1 models
            stream = stream_o1(messages, MODELS[model]['api_name'])
        
        output_chars = 0
        
        async def enhanced_stream():
            nonlocal output_chars
            yield f"data: {json.dumps({'type': 'cost', 'cost': estimated_cost})}\n\n"
            
            async for chunk in stream:
                if 'content' in chunk:
                    output_chars += len(chunk['content'])
                    # Update cost based on actual output length
                    final_cost = calculate_cost(model, total_input_chars, output_chars)
                    # Only yield new cost if it's significantly different
                    if abs(final_cost - estimated_cost) > 0.0001:
                        estimated_cost = final_cost
                        yield f"data: {json.dumps({'type': 'cost', 'cost': final_cost})}\n\n"
                yield f"data: {json.dumps(chunk)}\n\n"

        # Update user credits with final cost after streaming
        if not user.get("is_free_tier"):
            final_cost = calculate_cost(model, total_input_chars, output_chars)
            user["credits"] -= final_cost
            update_user(user)
            log_usage(user["id"], model, final_cost, total_input_chars + output_chars, "chat")

        return StreamingResponse(
            enhanced_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error in stream_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

def calculate_cost(model: str, input_chars: int, output_chars: int = 0) -> float:
    """Calculate cost based on input and output characters for a given model"""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    
    model_costs = MODELS[model]['cost']
    input_tokens = input_chars / 4  # Approximate tokens
    output_tokens = output_chars / 4
    
    return (input_tokens * model_costs['input'] + output_tokens * model_costs['output']) / 1000

def log_usage(user_id: str, model: str, cost: float, chars: int, feature: str = "solve"):
    with open(USAGE_FILE, 'r+') as f:
        usage = json.load(f)
        usage["items"].append({
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "cost": cost,
            "chars": chars,
            "feature": feature
        })
        f.seek(0)
        json.dump(usage, f)
        f.truncate()


@app.post("/api/solve")
async def solve_question(request: SolveRequest, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        # Calculate cost including context if provided
        total_chars = len(request.text)
        if request.context:
            total_chars += len(json.dumps(request.context))
        cost = calculate_cost(request.model, total_chars, bool(request.context))

        if not user.get("is_free_tier") and user["credits"] < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        # Prepare prompt with context
        prompt = request.text
        if request.context:
            prompt = f"Context: {json.dumps(request.context)}\nQuestion: {request.text}"

        if request.model.startswith("gpt"):
            # OpenAI API call
            system_prompt = """You are an expert at answering questions. Provide your response as a valid JSON object with exactly these fields:
            - 'answer' (a concise, clear answer)
            - 'explanation' (a brief, helpful explanation)
            - 'confidence' (a number 0-100)
            - 'next_step' (optional text to identify next/continue button if applicable)
            Example: {"answer": "7", "explanation": "To find the area, multiply length (3) by width (2): 3 × 2 = 6", "confidence": 95, "next_step": "Continue"}"""

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
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="OpenAI API error")
                
                result = json.loads(response.json()["choices"][0]["message"]["content"])

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
                            "content": f"Answer this question and format as JSON with 'answer', 'explanation', 'confidence', and optional 'next_step' fields: {prompt}"
                        }]
                    }
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="Anthropic API error")

                result = json.loads(response.json()["content"][0]["text"])

        # Update user credits and log usage
        if not user.get("is_free_tier"):
            user["credits"] -= cost
            update_user(user)
        
        log_usage(user["id"], request.model, cost, total_chars)

        return {
            "response": result,
            "cost": cost,
            "cached": False
        }

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

@app.post("/api/chat")
async def chat(request: ChatRequest, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        cost = calculate_cost(request.model, len(request.message))
        if not user.get("is_free_tier") and user["credits"] < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        # Format chat history for API
        formatted_history = []
        for msg in request.history:
            formatted_history.append({
                "role": "user" if msg["type"] == "user" else "assistant",
                "content": msg["content"]
            })

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
                        "messages": formatted_history + [{"role": "user", "content": request.message}]
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="OpenAI API error")
                
                answer = response.json()["choices"][0]["message"]["content"]

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
                        "max_tokens": 500,
                        "messages": formatted_history + [{"role": "user", "content": request.message}]
                    }
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="Anthropic API error")

                answer = response.json()["content"][0]["text"]

        # Update user credits and log usage
        if not user.get("is_free_tier"):
            user["credits"] -= cost
            update_user(user)
        
        log_usage(user["id"], request.model, cost, len(request.message), "chat")

        return {
            "response": answer,
            "cost": cost
        }

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest, auth: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        # Calculate cost (images cost more)
        base_chars = len(request.image) + (len(request.question) if request.question else 0)
        cost = calculate_cost(request.model, base_chars) * 1.5

        if not user.get("is_free_tier") and user["credits"] < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        # Process image and question using GPT-4 Vision
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEYS['OPENAI']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": request.question or "What's in this image?"},
                                {"type": "image_url", "image_url": {"url": request.image}}
                            ]
                        }
                    ],
                    "max_tokens": 300
                }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Vision API error")

            result = response.json()["choices"][0]["message"]["content"]

        # Update user credits and log usage
        if not user.get("is_free_tier"):
            user["credits"] -= cost
            update_user(user)
        
        log_usage(user["id"], "gpt4-vision", cost, base_chars, "image")

        return {
            "response": result,
            "cost": cost
        }

    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)  # Changed port to 
