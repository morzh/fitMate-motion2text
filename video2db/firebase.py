from pathlib import Path
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account.
cred = credentials.Certificate('fitmateapp.json')
databaseURL = "https://fitmateapp-a9a5e-default-rtdb.europe-west1.firebasedatabase.app"
# app = firebase_admin.initialize_app(cred, {'databaseURL': databaseURL})
app = firebase_admin.initialize_app(cred)
db = firestore.client(app)

COLLECTION = u'videos'

 
def upload_video(user_id: str, video_path: Path):
    data = {
        u'format': u'{}'.format(video_path.suffix),
        # u'video': open(video_path, 'rb'),
    }
    doc_ref = db.collection(COLLECTION).document(u'{}:{}'.format(user_id, video_path.stem))
    doc_ref.set(data)
# FailedPrecondition('The Cloud Firestore API is not available for Firestore in Datastore Mode database projects/fitmateapp-a9a5e/databases/(default).')
def download_video():
    users_ref = db.collection(COLLECTION)
    docs = users_ref.stream()

    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')


if __name__ == '__main__':
    video_path = Path(
        "/home/ubuntu/PycharmProjects/FitMate/fitMate-motion2text/action_recognition/dataset/from_team/back_and_delta/IMG_4392.mov")
    upload_video("test_user", video_path)
    download_video()
