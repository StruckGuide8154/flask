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
MODELS = {
    # Gemini Models
    "gemini-pro": {
        "input": 1.25*40, 
        "output": 5.00*40,
        "context_window": 1000000,
        "max_output_tokens": 2048,
        "features": ["text", "code", "vision", "audio"],
        "supports_stream": True
    },
    "gemini-pro-2": {
        "input": 2.50*40, 
        "output": 10.00*40,
        "context_window": 2000000,
        "max_output_tokens": 4096,
        "features": ["text", "code", "vision", "audio", "video"],
        "supports_stream": True
    },
    "gemini-flash": {
        "input": 0.075*40, 
        "output": 0.30*40,
        "context_window": 128000,
        "max_output_tokens": 2048,
        "features": ["text", "vision"],
        "supports_stream": True
    },

    # OpenAI Models
    "gpt4o": {
        "input": 0.0025*40, 
        "output": 0.00125*40,
        "context_window": 128000,
        "max_output_tokens": 16384,
        "features": ["text", "vision", "code"],
        "supports_stream": True
    },
    "gpt4o-mini": {
        "input": 0.00015*40, 
        "output": 0.000075*40,
        "context_window": 128000,
        "max_output_tokens": 16384,
        "features": ["text", "vision"],
        "supports_stream": True
    },

    # Claude Models
    "claude-haiku": {
        "input": 0.003*40, 
        "output": 0.00375*40,
        "context_window": 200000,
        "max_output_tokens": 4096,
        "features": ["text", "vision", "code"],
        "supports_stream": True
    },
    "claude-opus": {
        "input": 0.015*40, 
        "output": 0.075*40,
        "context_window": 200000,
        "max_output_tokens": 8192,
        "features": ["text", "vision", "code", "audio"],
        "supports_stream": True
    },
    "claude-sonnet": {
        "input": 0.008*40, 
        "output": 0.024*40,
        "context_window": 200000,
        "max_output_tokens": 8192,
        "features": ["text", "vision", "code"],
        "supports_stream": True
    }
}

# Initialize API clients
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "default-key"))
claude_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY", "default-key"))

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_tokens: Optional[int] = 0
    total_cost: float

class FileContent(BaseModel):
    mime_type: str
    data: str
    token_count: int

async def process_file(file: UploadFile) -> FileContent:
    """Process uploaded file and return content with token count"""
    content = await file.read()
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0]

    if mime_type.startswith('image/'):
        # Process image
        img = Image.open(io.BytesIO(content))
        width, height = img.size
        token_count = 258  # Base image token cost

        # Add resolution-based tokens
        token_count += (width * height) // (512 * 512) * 85

    elif mime_type.startswith('audio/'):
        # Process audio
        container = av.open(io.BytesIO(content))
        duration = float(container.duration) / av.time_base
        token_count = int(duration * 25)  # 25 tokens per second

    elif mime_type.startswith('video/'):
        # Process video
        container = av.open(io.BytesIO(content))
        duration = float(container.duration) / av.time_base
        token_count = int(duration * 300)  # 300 tokens per second includes frames/audio

    else:
        # Text and other files
        text_content = content.decode('utf-8', errors='ignore')
        token_count = len(text_content.split()) * 2  # Rough estimation

    return FileContent(
        mime_type=mime_type,
        data=base64.b64encode(content).decode(),
        token_count=token_count
    )

async def calculate_context_length(
    text: str,
    files: List[FileContent],
    chat_history: List[Dict] = None
) -> Dict[str, int]:
    """Calculate complete context length including history"""
    tokens = {
        "input_tokens": len(text.split()) * 2,  # Basic estimation
        "file_tokens": sum(f.token_count for f in files),
        "history_tokens": 0
    }

    if chat_history:
        for msg in chat_history:
            tokens["history_tokens"] += len(str(msg).split()) * 2

    tokens["total"] = sum(tokens.values())
    return tokens

@app.post("/api/ai/solve")
async def solve_with_model(
    request: Request,
    files: List[UploadFile] = File(None),
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """Main endpoint for model interaction with full feature support"""
    try:
        # Verify token and get user
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        # Parse request
        data = await request.json()
        model = data.get("model", "gpt4o-mini")
        content = data.get("text", "")
        stream = data.get("stream", False)
        chat_history = data.get("history", [])
        code_execution = data.get("code_execution", False)

        # Process files
        processed_files = []
        if files:
            processed_files = [await process_file(file) for file in files]

        # Calculate context length
        context = await calculate_context_length(content, processed_files, chat_history)

        # Check context window limits
        if context["total"] > MODELS[model]["context_window"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Context length {context['total']} exceeds model limit of {MODELS[model]['context_window']}"
            )

        # Calculate cost
        model_config = MODELS[model]
        cost = (
            (context["input_tokens"] + context["file_tokens"] + context["history_tokens"]) 
            * model_config["input"]
        ) + (context["total"] * 1.5 * model_config["output"])

        # Check credits
        if not user.get("is_free_tier") and user["credits"] < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        # Generate response based on model type
        if model.startswith("gpt"):
            response = await handle_openai(
                model, content, processed_files, stream, chat_history, code_execution
            )
        elif model.startswith("claude"):
            response = await handle_claude(
                model, content, processed_files, stream, chat_history, code_execution
            )
        else:
            response = await handle_gemini(
                model, content, processed_files, stream, chat_history, code_execution
            )

        # Update user credits and log usage
        if not user.get("is_free_tier"):
            user["credits"] -= cost
            update_user(user)

        log_usage(
            user["id"],
            model,
            context["total"],
            cost,
            len(content),
            len(processed_files)
        )

        if stream:
            return StreamingResponse(
                response,
                media_type="text/event-stream"
            )
        return response

    except Exception as e:
        logger.error(f"Error in solve_with_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_openai(
    model: str,
    content: str,
    files: List[FileContent],
    stream: bool,
    history: List[Dict],
    code_execution: bool
):
    """Enhanced OpenAI handler with all features"""
    messages = []

    # Add history
    if history:
        messages.extend(history)

    # Add file contents
    for file in files:
        if file.mime_type.startswith(('image/', 'video/')):
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file.mime_type};base64,{file.data}"
                        }
                    }
                ]
            })

    # Add text content
    messages.append({"role": "user", "content": content})

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEYS['OPENAI']}"},
            json={
                "model": "gpt-4-vision-preview" if any(f.mime_type.startswith(('image/', 'video/')) for f in files) else model,
                "messages": messages,
                "stream": stream,
                "max_tokens": MODELS[model]["max_output_tokens"],
                "temperature": 0.7,
                "tools": [{"type": "code_interpreter"}] if code_execution else None
            }
        )

        if stream:
            return response.aiter_lines()

        return response.json()

async def handle_claude(
    model: str,
    content: str,
    files: List[FileContent],
    stream: bool,
    history: List[Dict],
    code_execution: bool
):
    """Enhanced Claude handler with all features"""
    messages = []

    # Add history
    if history:
        messages.extend(history)

    # Process files
    for file in files:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": file.mime_type.split('/')[0],
                    "source": {
                        "type": "base64",
                        "media_type": file.mime_type,
                        "data": file.data
                    }
                }
            ]
        })

    # Add text content
    messages.append({
        "role": "user",
        "content": content
    })

    response = await claude_client.messages.create(
        model=model,
        max_tokens=MODELS[model]["max_output_tokens"],
        messages=messages,
        stream=stream,
        tools=[{"type": "code_interpreter"}] if code_execution else None
    )

    if stream:
        async for chunk in response:
            yield chunk
    else:
        yield response.text

async def handle_gemini(
    model: str,
    content: str,
    files: List[FileContent],
    stream: bool,
    history: List[Dict],
    code_execution: bool
):
    """Enhanced Gemini handler with all features"""

    # Configure model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": MODELS[model]["max_output_tokens"],
    }

    tools = None
    if code_execution:
        tools = ["code_execution"]

    genai_model = genai.GenerativeModel(
        model_name=f"gemini-{model}",
        generation_config=generation_config,
        tools=tools
    )

    # Start chat if history exists
    if history:
        chat = genai_model.start_chat(history=history)

    # Process contents
    contents = []
    for file in files:
        contents.append({
            "mime_type": file.mime_type,
            "data": file.data
        })

    # Add text content
    contents.append(content)

    # Generate response
    if history:
        response = await chat.send_message(
            contents,
            stream=stream
        )
    else:
        response = await genai_model.generate_content(
            contents,
            stream=stream
        )

    if stream:
        async for chunk in response:
            yield chunk
    else:
        yield response.text


# Add this to your FastAPI routes
@app.get("/ai")
async def serve_expansionai_interface():
    """Serve the ExpansionAI interface"""
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ExpansionAI | Advanced Intelligence Interface</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism-tomorrow.min.css">
    <style>
        :root {
            --gold-primary: #BF9B30;
            --gold-secondary: #D4AF37;
            --gold-highlight: #FFD700;
            --black-primary: #0A0A0A;
            --black-secondary: #141414;
            --black-highlight: #1A1A1A;
            --gradient-gold: linear-gradient(135deg, var(--gold-primary), var(--gold-secondary));
        }
        * { font-family: 'Inter', system-ui, sans-serif; }
        
        body { background: var(--black-primary); color: #ffffff; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--black-secondary); }
        ::-webkit-scrollbar-thumb { 
            background: var(--gold-primary); 
            border-radius: 3px;
        }
        .sidebar {
            background: var(--black-secondary);
            border-right: 1px solid rgba(191, 155, 48, 0.2);
            box-shadow: 2px 0 20px rgba(0, 0, 0, 0.3);
        }
        .chat-container {
            background: radial-gradient(circle at top, #1a1a1a, var(--black-primary));
        }
        .message-input {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(191, 155, 48, 0.2);
            backdrop-filter: blur(10px);
        }
        .gold-button {
            background: var(--gradient-gold);
            color: var(--black-primary);
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .model-selector {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(191, 155, 48, 0.2);
            color: var(--gold-primary);
        }
        .ai-message {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(191, 155, 48, 0.1);
            backdrop-filter: blur(10px);
        }
        .user-message {
            background: rgba(191, 155, 48, 0.05);
            border: 1px solid rgba(191, 155, 48, 0.2);
        }
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 0.5rem;
        }
        .typing-dot {
            width: 4px;
            height: 4px;
            background: var(--gold-primary);
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-4px); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in { animation: fadeIn 0.3s ease forwards; }
    </style>
</head>
<body class="overflow-hidden">
    <div class="flex h-screen">
        <!-- Sidebar -->
        <div class="sidebar w-80 flex flex-col">
            <div class="p-6 flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-gold-primary to-gold-secondary flex items-center justify-center">
                    <svg class="w-6 h-6 text-black" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M13.5 3.5L12 2l-1.5 1.5L9 2 7.5 3.5 6 2 4.5 3.5 3 2v20l1.5-1.5L6 22l1.5-1.5L9 22l1.5-1.5L12 22l1.5-1.5L15 22l1.5-1.5L18 22l1.5-1.5L21 22V2l-1.5 1.5L18 2l-1.5 1.5L15 2l-1.5 1.5z"/>
                    </svg>
                </div>
                <h1 class="text-xl font-bold bg-gradient-to-r from-gold-primary to-gold-highlight bg-clip-text text-transparent">
                    ExpansionAI
                </h1>
            </div>
            <div class="flex-1 overflow-y-auto p-4 space-y-4" id="chat-list">
                <!-- Chat history will be populated here -->
            </div>
            <div class="p-4 m-4 rounded-xl bg-black-highlight border border-gold-600/20">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 rounded-full bg-gradient-to-br from-gold-primary to-gold-secondary flex items-center justify-center text-black font-bold user-initial">
                        H
                    </div>
                    <div>
                        <div class="font-medium user-name">User</div>
                        <div class="text-xs text-gold-primary user-plan">Professional Plan</div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Main Chat Area -->
        <div class="flex-1 flex flex-col">
            <div class="p-4 border-b border-gold-600/10 flex items-center justify-between">
                <div class="flex items-center space-x-4">
                    <select class="model-selector p-2 rounded-lg" id="model-select">
                        <optgroup label="Fast Models">
                            <option value="gemini-flash-8b">Gemini Flash 8B</option>
                            <option value="gemini-flash">Gemini Flash</option>
                            <option value="gemini-pro">Gemini Pro</option>
                        </optgroup>
                        <optgroup label="Standard Models">
                            <option value="gpt4o-mini">GPT-4 Mini</option>
                            <option value="gpt4o">GPT-4 Opus</option>
                        </optgroup>
                        <optgroup label="Advanced Models">
                            <option value="claude-haiku">Claude Haiku</option>
                            <option value="claude-opus">Claude Opus</option>
                            <option value="claude-sonnet">Claude Sonnet</option>
                        </optgroup>
                    </select>
                </div>
                <div class="flex items-center space-x-3">
                    <button class="gold-button px-4 py-2 rounded-lg" id="clear-chat">Clear Chat</button>
                    <button class="gold-button px-4 py-2 rounded-lg" id="export-chat">Export Chat</button>
                </div>
            </div>
            <div class="flex-1 overflow-y-auto p-6 chat-container" id="chat-messages">
                <!-- Messages will be added here -->
            </div>
            <div class="p-6 border-t border-gold-600/10">
                <div class="max-w-4xl mx-auto">
                    <div id="file-previews" class="flex flex-wrap gap-2 mb-4">
                        <!-- File previews will appear here -->
                    </div>
                    
                    <div class="flex items-end space-x-4">
                        <div class="flex-1 relative">
                            <textarea 
                                id="message-input"
                                class="message-input w-full rounded-xl py-3 px-4 text-white resize-none"
                                placeholder="Send a message..."
                                rows="3"
                            ></textarea>
                            
                            <div class="absolute bottom-3 right-3 flex space-x-2">
                                <button id="upload-btn" class="p-2 hover:bg-black-highlight rounded-lg text-gold-primary">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"/>
                                    </svg>
                                </button>
                                <button id="image-btn" class="p-2 hover:bg-black-highlight rounded-lg text-gold-primary">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                                    </svg>
                                </button>
                            </div>
                        </div>
                        <button id="send-btn" class="gold-button p-3 rounded-xl flex items-center space-x-2">
                            <span>Send</span>
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
    
    <script>
        class ExpansionAIChat {
            constructor() {
                this.messageInput = document.getElementById('message-input');
                this.sendButton = document.getElementById('send-btn');
                this.chatMessages = document.getElementById('chat-messages');
                this.modelSelect = document.getElementById('model-select');
                this.uploadBtn = document.getElementById('upload-btn');
                this.imageBtn = document.getElementById('image-btn');
                this.clearChat = document.getElementById('clear-chat');
                this.exportChat = document.getElementById('export-chat');
                this.filePreviews = document.getElementById('file-previews');
                
                this.files = [];
                this.messages = [];
                this.isStreaming = false;
                
                this.initializeEventListeners();
                this.loadUserInfo();
                this.showWelcomeMessage();
            }
            initializeEventListeners() {
                this.sendButton.addEventListener('click', () => this.sendMessage());
                this.messageInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
                this.uploadBtn.addEventListener('click', () => this.handleFileUpload());
                this.imageBtn.addEventListener('click', () => this.handleImageUpload());
                this.clearChat.addEventListener('click', () => this.clearAllMessages());
                this.exportChat.addEventListener('click', () => this.exportChatHistory());
                // File drag and drop
                document.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });
                document.addEventListener('drop', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const files = Array.from(e.dataTransfer.files);
                    this.handleFiles(files);
                });
            }
            async loadUserInfo() {
                try {
                    const response = await fetch('/api/user/info', {
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        }
                    });
                    if (response.ok) {
                        const userData = await response.json();
                        document.querySelector('.user-name').textContent = userData.name;
                        document.querySelector('.user-initial').textContent = userData.name[0];
                        document.querySelector('.user-plan').textContent = userData.plan;
                    }
                } catch (error) {
                    console.error('Error loading user info:', error);
                }
            }
            showWelcomeMessage() {
                const welcomeMessage = {
                    role: 'assistant',
                    content: `Welcome to ExpansionAI! I can help you with:
                    - Complex code analysis and generation
                    - Image and video understanding
                    - Data analysis and visualization
                    - Natural language tasks`,
                    model: 'gemini-pro'
                };
                this.addMessage(welcomeMessage);
            }
            async sendMessage() {
                const content = this.messageInput.value.trim();
                if (!content && this.files.length === 0) return;
                const userMessage = {
                    role: 'user',
                    content,
                    files: this.files
                };
                this.addMessage(userMessage);
                this.messageInput.value = '';
                this.showTypingIndicator();
                const formData = new FormData();
                formData.append('text', content);
                formData.append('model', this.modelSelect.value);
                formData.append('stream', 'true');
                
                this.files.forEach(file => {
                    formData.append('files', file);
                });
                try {
                    const response = await fetch('/api/ai/solve', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        },
                        body: formData
                    });
                    if (!response.ok) {
                        throw new Error('API request failed');
                    }
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let accumulatedResponse = '';
                    while (true) {
                        const {value, done} = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        accumulatedResponse += chunk;
                        // Update the ongoing message
                        this.updateStreamingMessage(accumulatedResponse);
                    }
                    // Finalize the message
                    this.finalizeStreamingMessage(accumulatedResponse);
                    
                } catch (error) {
                    console.error('Error sending message:', error);
                    this.showError('Failed to send message');
                } finally {
                    this.hideTypingIndicator();
                    this.clearFiles();
                }
            }
            addMessage(message) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `${message.role}-message rounded-xl p-4 max-w-4xl mx-auto mb-6 fade-in`;
                
                const isUser = message.role === 'user';
                
                let html = `
                    <div class="flex space-x-4 ${isUser ? 'flex-row-reverse' : ''}">
                        <div class="w-8 h-8 rounded-lg ${isUser ? 'bg-blue-600' : 'bg-gradient-to-br from-gold-primary to-gold-secondary'} flex-shrink-0 flex items-center justify-center">
                            ${isUser ? '<span class="text-white font-medium user-initial">U</span>' : 
                            '<svg class="w-5 h-5 text-black" viewBox="0 0 24 24" fill="currentColor"><path d="M13.5 3.5L12 2l-1.5 1.5L9 2 7.5 3.5 6 2 4.5 3.5 3 2v20l1.5-1.5L6 22l1.5-1.5L9 22l1.5-1.5L12 22l1.5-1.5L15 22l1.5-1.5L18 22l1.5-1.5L21 22V2l-1.5 1.5L18 2l-1.5 1.5L15 2l-1.5 1.5z"/></svg>'}
                        </div>
                        <div class="flex-1">
                            <div class="flex items-center space-x-2 ${isUser ? 'justify-end' : ''}">
                                <span class="font-medium">${isUser ? 'You' : 'ExpansionAI'}</span>
                                ${!isUser && message.model ? `<span class="text-xs bg-gold-primary/10 text-gold-primary px-2 py-1 rounded">${message.model}</span>` : ''}
                            </div>`;
                // Add file previews if present
                if (message.files && message.files.length > 0) {
                    html += '<div class="flex flex-wrap gap-2 mt-2">';
                    message.files.forEach(file => {
                        html += this.createFilePreviewHTML(file);
                    });
                    html += '</div>';
                }
                // Add message content
                const formattedContent = this.formatMessageContent(message.content);
                html += `<div class="mt-2 prose prose-invert">${formattedContent}</div>`;
                
                html += '</div></div>';
                
                messageDiv.innerHTML = html;
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
                // Initialize code highlighting if needed
                if (messageDiv.querySelector('pre code')) {
                    Prism.highlightAllUnder(messageDiv);
                }
            }
            formatMessageContent(content) {
                if (!content) return '';
                
                // Convert markdown to HTML
                let html = marked.parse(content);
                // Special handling for code blocks
                html = html.replace(/<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g, 
                    (_, lang, code) => `
                    <div class="code-block relative">
                        <div class="code-actions absolute top-2 right-2 opacity-0 transition-opacity">
                            <button class="p-1 hover:bg-black-highlight rounded" onclick="navigator.clipboard.writeText(this.parentElement.parentElement.querySelector('code').textContent)">
                                <svg class="w-4 h-4 text-gold-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                                </svg>
                            </button>
                        </div>
                        <pre><code class="language-${lang}">${code}</code></pre>
                    </div>`
                );
                return html;
            }
            showTypingIndicator() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'typing-indicator';
                typingDiv.innerHTML = `
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                `;
                this.chatMessages.appendChild(typingDiv);
                this.scrollToBottom();
            }
            hideTypingIndicator() {
                const indicator = this.chatMessages.querySelector('.typing-indicator');
                if (indicator) {
                    indicator.remove();
                }
            }
            updateStreamingMessage(content) {
                let streamingMsg = this.chatMessages.querySelector('.streaming-message');
                if (!streamingMsg) {
                    streamingMsg = document.createElement('div');
                    streamingMsg.className = 'assistant-message rounded-xl p-4 max-w-4xl mx-auto mb-6 fade-in streaming-message';
                    this.chatMessages.appendChild(streamingMsg);
                }
                
                streamingMsg.innerHTML = this.formatMessageContent(content);
                this.scrollToBottom();
            }
            finalizeStreamingMessage(content) {
                const streamingMsg = this.chatMessages.querySelector('.streaming-message');
                if (streamingMsg) {
                    streamingMsg.classList.remove('streaming-message');
                }
            }
            async handleFiles(files) {
                const allowedTypes = {
                    'image': ['jpeg', 'png', 'gif', 'webp'],
                    'video': ['mp4', 'webm', 'ogg'],
                    'audio': ['mp3', 'wav', 'ogg'],
                    'application': ['pdf', 'json', 'txt']
                };
                for (const file of files) {
                    const [type, subtype] = file.type.split('/');
                    if (allowedTypes[type]?.includes(subtype)) {
                        this.files.push(file);
                        this.addFilePreview(file);
                    }
                }
            }
            addFilePreview(file) {
                const preview = document.createElement('div');
                preview.className = 'file-preview p-2 bg-black-highlight rounded flex items-center space-x-2';
                
                const icon = this.getFileIcon(file.type);
                preview.innerHTML = `
                    ${icon}
                    <span class="text-sm truncate max-w-xs">${file.name}</span>
                    <button class="text-gold-primary hover:text-gold-secondary">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                    </button>
                `;
                preview.querySelector('button').onclick = () => {
                    this.removeFile(file);
                    preview.remove();
                };
                this.filePreviews.appendChild(preview);
            }
            getFileIcon(type) {
                const [mainType] = type.split('/');
                const icons = {
                    'image': `<svg class="w-5 h-5 text-gold-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                    </svg>`,
                    'video': `<svg class="w-5 h-5 text-gold-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                    </svg>`,
                    'audio': `<svg class="w-5 h-5 text-gold-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"/>
                    </svg>`
                };
                
                return icons[mainType] || `<svg class="w-5 h-5 text-gold-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/>
                </svg>`;
            }
            removeFile(file) {
                const index = this.files.indexOf(file);
                if (index > -1) {
                    this.files.splice(index, 1);
                }
            }
            clearFiles() {
                this.files = [];
                this.filePreviews.innerHTML = '';
            }
            scrollToBottom() {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }
            clearAllMessages() {
                this.chatMessages.innerHTML = '';
                this.showWelcomeMessage();
            }
            exportChatHistory() {
                const messages = Array.from(this.chatMessages.children).map(msg => {
                    return {
                        role: msg.classList.contains('user-message') ? 'user' : 'assistant',
                        content: msg.querySelector('.prose')?.textContent || '',
                        timestamp: new Date().toISOString()
                    };
                });
                const blob = new Blob([JSON.stringify(messages, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `chat-history-${new Date().toISOString()}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
            showError(message) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded shadow-lg fade-in';
                errorDiv.textContent = message;
                document.body.appendChild(errorDiv);
                setTimeout(() => errorDiv.remove(), 3000);
            }
        }
        // Initialize the chat application
        const chat = new ExpansionAI();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)



@app.post("/api/ai/upload")
async def upload_file(
    file: UploadFile = File(...),
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """Handle file uploads with token counting"""
    try:
        processed_file = await process_file(file)
        return {
            "success": True,
            "file_info": {
                "name": file.filename,
                "type": processed_file.mime_type,
                "token_count": processed_file.token_count
            }
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/models")
async def get_models(
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """Get available models and their capabilities"""
    return {
        "models": MODELS
    }

@app.get("/api/ai/usage/{user_id}")
async def get_user_usage(
    user_id: str,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """Get detailed usage statistics for a user"""
    try:
        payload = jwt.decode(auth.credentials,SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user or not user["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        # Read usage from database
        with open(USAGE_FILE) as f:
            usage_data = json.load(f)
            user_usage = [item for item in usage_data["items"] if item["user_id"] == user_id]

        # Calculate aggregated statistics
        total_tokens = sum(item["tokens"] for item in user_usage)
        total_cost = sum(item["cost"] for item in user_usage)
        model_usage = {}

        for item in user_usage:
            model = item["model"]
            if model not in model_usage:
                model_usage[model] = {
                    "count": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            model_usage[model]["count"] += 1
            model_usage[model]["tokens"] += item["tokens"]
            model_usage[model]["cost"] += item["cost"]

        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "model_usage": model_usage,
            "detailed_usage": user_usage
        }

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def log_usage(user_id: str, model: str, tokens: int, cost: float, chars: int, files: int = 0):
    """Enhanced usage logging with file tracking"""
    with open(USAGE_FILE) as f:
        usage = json.load(f)

    usage["items"].append({
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "tokens": tokens,
        "cost": cost,
        "chars": chars,
        "files_processed": files
    })

    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f)

@app.get("/api/ai/context_window/{model}")
async def get_context_window(
    model: str,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """Get context window information for a model"""
    try:
        if model not in MODELS:
            raise HTTPException(status_code=404, detail="Model not found")

        return {
            "model": model,
            "context_window": MODELS[model]["context_window"],
            "max_output_tokens": MODELS[model]["max_output_tokens"],
            "features": MODELS[model]["features"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/cache")
async def create_cache(
    request: Dict,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """Create a context cache for large files/contexts"""
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        cache_id = f"cache_{user['id']}_{int(time.time())}"

        # Store in Redis with 1 hour expiration
        redis_client.setex(
            cache_id,
            3600,  # 1 hour
            json.dumps({
                "user_id": user["id"],
                "content": request.get("content", ""),
                "files": request.get("files", []),
                "created_at": datetime.utcnow().isoformat()
            })
        )

        return {
            "cache_id": cache_id,
            "expires_in": 3600
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/cache/{cache_id}")
async def get_cache(
    cache_id: str,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """Retrieve cached content"""
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=["HS256"])
        user = get_user(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        cached = redis_client.get(cache_id)
        if not cached:
            raise HTTPException(status_code=404, detail="Cache not found")

        cached_data = json.loads(cached)
        if cached_data["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized to access this cache")

        return cached_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Frontend helper components
@app.get("/components/ai-chat.js")
async def get_chat_component():
    """Serve the AI chat component"""
    return FileResponse(
        "static/components/ai-chat.js",
        media_type="application/javascript"
    )

@app.get("/components/file-upload.js")
async def get_file_upload_component():
    """Serve the file upload component"""
    return FileResponse(
        "static/components/file-upload.js",
        media_type="application/javascript"
    )


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
