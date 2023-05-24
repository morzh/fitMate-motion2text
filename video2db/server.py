import os
from dotenv import load_dotenv
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, status
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import FileTarget, ValueTarget
from streaming_form_data.validators import MaxSizeValidator
import streaming_form_data
from starlette.requests import ClientDisconnect

load_dotenv()

MAX_FILE_SIZE = 1024 * 1024 * 1024 * 4  # = 4GB
MAX_REQUEST_BODY_SIZE = MAX_FILE_SIZE + 1024

app = FastAPI()

@app.post('/upload')
async def upload(request: Request):
    body_validator = MaxBodySizeValidator(MAX_REQUEST_BODY_SIZE)
    filename = request.headers.get('Filename')
    video_dir = Path("./videos")
    video_dir.mkdir(exist_ok=True)
    available_tokens = os.environ.get("CLIENT_TOKEN")

    if not filename:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail='Filename header is missing')

    try:
        filepath = video_dir/os.path.basename(filename)
        file_ = FileTarget(str(filepath), validator=MaxSizeValidator(MAX_FILE_SIZE))
        data = ValueTarget()
        token = data.value.decode()
        if token not in available_tokens:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail='token is not valid')
        parser = StreamingFormDataParser(headers=request.headers)
        parser.register('file', file_)
        parser.register('data', data)

        async for chunk in request.stream():
            body_validator(chunk)
            parser.data_received(chunk)
    except ClientDisconnect:
        print("Client Disconnected")
    except MaxBodySizeException as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f'Maximum request body size limit ({MAX_REQUEST_BODY_SIZE} bytes) exceeded ({e.body_len} bytes read)')
    except streaming_form_data.validators.ValidationError:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f'Maximum file size limit ({MAX_FILE_SIZE} bytes) exceeded')
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail='There was an error uploading the file')

    if not file_.multipart_filename:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail='File is missing')

    print(data.value.decode())
    print(file_.multipart_filename)

    return {"message": f"Successfuly uploaded {filename}"}


class MaxBodySizeException(Exception):
    def __init__(self, body_len: int):
        self.body_len = body_len


class MaxBodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(body_len=self.body_len)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7777, log_level="info")

