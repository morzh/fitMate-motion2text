import os
import httpx
import time

url = 'https://0.0.0.0:7777/upload'
url = 'http://0.0.0.0:7777/video_info'
with httpx.Client() as client:
    start = time.time()
    r = client.get(url, headers={"link": "https://youtu.be/dQw4w9WgXcQ"})

p=0
def send_video(path):
    files = {'file': open(path, 'rb')}
    headers = {'Filename': 'test_video.mov'}
    data = {'token': os.environ.get("CLIENT_TOKEN")}

    with httpx.Client() as client:
        start = time.time()
        r = client.post(url, data=data, files=files, headers=headers)
        end = time.time()
        print(f'Time elapsed: {end - start}s')
        print(r.status_code, r.json(), sep=' ')


if __name__ == '__main__':
    video_path = "/home/ubuntu/PycharmProjects/FitMate/fitMate-motion2text/action_recognition/dataset/from_team/back_and_delta/IMG_4392.mov"
    send_video(video_path)