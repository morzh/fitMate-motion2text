import os
from dotenv import load_dotenv
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request

load_dotenv()

app = FastAPI()

database = {
    "https://youtu.be/dQw4w9WgXcQ": {
        "chapters": {5: ["Хорошее начало", "бомба"],
                     100: ["Тут тоже всё ок", "Но можно лучше"],
                     150: []}
    }
}

tags_metadata = [
    {
        "name": "videos",
        "description": "Get request will return list of video urls on youtube.",
    },
    {
        "name": "video_info",
        "description": "Get request with header {'link': youtube_video_link} will return"
                       "video info.",
        "parameters": {"link": "youtu.be"}

    },
]

app = FastAPI(title="FitMate database API.",
              openapi_tags=tags_metadata)


@app.get('/videos', tags=["videos"])
async def get_videos():
    list_videos = list(database.keys())
    return {"videos": list_videos}


@app.get('/video_info', tags=["video_info"])
async def get_video_info(request: Request):
    link = request.headers.get('link')
    info = database.get(link, "video was not found.")
    return {"videos": info}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7777, log_level="info")
