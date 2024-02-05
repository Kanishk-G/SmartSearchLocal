from typing import Union, AsyncIterable
from fastapi import FastAPI, Body, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from smart_search_chatbot import SmartSearchStreaming
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
directory = "../AGHPDFs"
chatbot = SmartSearchStreaming(directory=directory)

class FormData(BaseModel):
    user_input: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse('smart_search.html', {"request": request})


@app.post("/get")
def smart_search(message: FormData):
    answer = chatbot.query_no_stream(message.user_input)
    return StreamingResponse(answer, media_type="text/event-stream")

@app.get("/file/{file_name}")
async def read_file(file_name: str):
    # Specify the local directory where your files are stored.
    file_path = f"{directory}/{file_name}"
    return FileResponse(file_path)