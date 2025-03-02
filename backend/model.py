import torch
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from scipy.special import expit

from sqlalchemy import create_engine, Column, Integer, String, Boolean, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ✅ 모델 저장 경로 설정
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# (추가) DB MySQL 연결
DATABASE_URL = "mysql+pymysql://root:12345678@localhost/video_db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DB에 저장
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

class VideoVote(Base):
    __tablename__ = "video_votes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    video_id = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False)
    vote = Column(Boolean, nullable=False)
    created_at = Column(TIMESTAMP, server_default="CURRENT_TIMESTAMP")

Base.metadata.create_all(bind=engine)


# ✅ **BlazeFace 모델 다운로드 (MTCNN 대체)**
face_detector = MTCNN(keep_all=True, device=device)

# ✅ **EfficientNet 모델 로드**
net = EfficientNet.from_pretrained('efficientnet-b4').to(device)
net.eval()

# ✅ **데이터 전처리 설정**
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화 추가
])


# ✅ 얼굴 감지 (MTCNN 사용)
def detect_faces(video_path, frame_step=10):
    cap = cv2.VideoCapture(video_path)
    face_detected = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 100:
            break

        if frame_count % frame_step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, _ = face_detector.detect(rgb_frame)
            if faces is not None and len(faces) > 0:
                face_detected = True
                break

        frame_count += 1

    cap.release()
    return face_detected

# ✅ 얼굴이 있는 경우 AI 탐지 (EfficientNet)
def detect_fake_face(video_path, frame_step=15):
    cap = cv2.VideoCapture(video_path)
    face_scores = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 100:
            break

        if frame_count % frame_step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, _ = face_detector.detect(rgb_frame)

            if faces is not None and len(faces) > 0:
                x1, y1, x2, y2 = map(int, faces[0])

                # ✅ 얼굴 크기 검증 (0이거나 음수면 기본 크기로 리사이즈)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    print(f"🚨 잘못된 얼굴 크기 감지 (x1={x1}, y1={y1}, x2={x2}, y2={y2}) → 기본 크기 사용")
                    face = cv2.resize(rgb_frame, (224, 224))
                else:
                    face = rgb_frame[y1:y2, x1:x2]

                # ✅ 크롭된 얼굴이 정상적인지 추가 체크
                if face.shape[0] == 0 or face.shape[1] == 0:
                    print("🚨 얼굴 크롭 실패 → 기본 크기 사용")
                    face = cv2.resize(rgb_frame, (224, 224))

                # ✅ 변환 후 모델 입력
                face_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    score = net(face_tensor).cpu().numpy().flatten()
                    face_scores.append(expit(score[0]))

        frame_count += 1

    cap.release()
    return np.mean(face_scores) if face_scores else 0.5

# ✅ 얼굴이 없는 경우 AI 탐지 (Optical Flow + Edge Map)
def detect_fake_no_face(video_path, frame_step=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 100:
            break

        if frame_count % frame_step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, _ = face_detector.detect(rgb_frame)

            if faces is not None and len(faces) > 0:
                x1, y1, x2, y2 = map(int, faces[0])
                print(f"✅ 감지된 얼굴 좌표: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cap.release()
                    return True

        frame_count += 1

    cap.release()
    return False


# ✅ **최종 실행 함수**
def predict_video(video_path):
    print(f"🔍 분석 시작: {video_path}")

    if detect_faces(video_path):
        print("🔵 얼굴 감지됨 → 얼굴 기반 AI 탐지")
        score = detect_fake_face(video_path)
    else:
        print("🟠 얼굴 없음 → 비얼굴 AI 탐지")
        score = detect_fake_no_face(video_path)

    fscore = 1 - score
    print(f"🔹 Score for real video: {score:.6f}")
    print(f"🔹 Score for fake video: {fscore:.6f}")

    result = {}
    if score > 0.5:
        print("✅ 이 영상은 **REAL** 입니다.")
        result["message"] = "✅ 이 영상은 **REAL** 입니다."
    else:
        print("🚨 이 영상은 **FAKE** 입니다!")
        result["message"] = "🚨 이 영상은 **FAKE** 입니다!"

    result["real_score"] = float(score)
    result["fake_score"] = float(fscore)

    return result

