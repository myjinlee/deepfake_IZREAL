# 🔥 Required Libraries
import os
import torch
import numpy as np
import cv2
from math import log, exp
from pytubefix import YouTube
from torch.utils.data import Dataset, DataLoader
import mysql.connector
from torchvision import models, transforms
from PIL import Image
import warnings

# py 파일 import가 안되서 export함
__all__=["predict_and_print_results"]

# ✅ Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# ✅ Suppress Warnings
warnings.filterwarnings("ignore")

# 🔥 MySQL Connection Function
def get_mysql_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='12345678',
            database='video_db'
        )
        return conn
    except mysql.connector.Error as e:
        print(f"❌ MySQL Connection Failed: {e}")
        exit(1)

# ✅ Fetch User Votes from video_votes Table
def get_user_votes(video_id):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(vote) FROM video_votes WHERE video_id = %s", (video_id,))
    user_count, real_count = cursor.fetchone()
    conn.close()

    if user_count == 0:
        return 0.5, 0  # No votes → neutral value

    p_u = float(real_count) / float(user_count)
    return p_u, float(user_count)

# 🔥 Dynamic Weight Adjustment Function
def calculate_weighted_score(p, p_u, user_count):
    p = float(p)
    p_u = float(p_u)
    user_count = float(user_count)

    if 0.3 <= p <= 0.6:
        w1 = 1 / (1 + exp(-10 * (p - 0.5)))  # Sigmoid Weighting
        w2 = log(user_count + 1)  # Log Weighting
        weighted_score = (p * w1 + p_u * w2) / (w1 + w2)
        print("✅ Weighted Score Applied")
    else:
        weighted_score = p
        print("❎ No Weighting Applied (AI Score Used)")
    
    return weighted_score

# ✅ EfficientNet Feature Extractor
class EfficientNetFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = models.efficientnet_b0(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image):
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return features.cpu().numpy().flatten()

# ✅ No-Face AI Detection (EfficientNet Feature Similarity)
def detect_fake_no_face(video_path, frame_step=5):
    feature_extractor = EfficientNetFeatureExtractor(device)
    cap = cv2.VideoCapture(video_path)
    prev_features = None
    feature_similarities = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        features = feature_extractor.extract_features(frame)
        if prev_features is not None and features is not None:
            similarity = np.dot(prev_features, features) / (np.linalg.norm(prev_features) * np.linalg.norm(features))
            feature_similarities.append(similarity)
        prev_features = features
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_step)
    
    cap.release()
    
    if len(feature_similarities) == 0:
        print("⚠️ Feature Similarities Empty. Using Default (0.5)")
        return 0.5
    
    avg_similarity = np.mean(feature_similarities)
    return 1 - avg_similarity

# ✅ YouTube Video Download
def download_youtube_video(url, save_path="downloads"):
    os.makedirs(save_path, exist_ok=True)
    yt = YouTube(url)
    video_title = yt.title.replace(" ", "_")
    stream = yt.streams.filter(file_extension="mp4", res="720p").first()
    if stream is None:
        stream = yt.streams.get_highest_resolution()
    save_file = os.path.join(save_path, f"{video_title}.mp4")
    stream.download(output_path=save_path, filename=f"{video_title}.mp4")
    print(f"✅ Download Complete: {save_file}")
    return save_file

# ✅ Prediction & Results Display
def predict_and_print_results(video_id, video_path):
    print("get_user 중")
    p_u, user_count = get_user_votes(video_id)
    print("detect_fake_no_face 중")
    score = detect_fake_no_face(video_path)
    fscore = 1 - score
    print("weighted score 계산 중")
    weighted_score = calculate_weighted_score(score, p_u, user_count)

    result_str = "--------------------------------------------------\n"
    result_str += "🔍 AI Prediction Result\n"
    result_str += f"🔹 AI Score for real video: {score:.6f}\n"
    #result_str += f"🔹 AI Score for fake video: {fscore:.6f}\n"
    if 0.3 <= score <= 0.6:
        print("✅ User Votes Considered ({int(user_count)} votes)\n")
        print( f"🔹 User Vote-Based Real Ratio: {p_u:.2f}\n")
        print("✅ Weighted Score Applied → AI & User Votes Combined\n")
    else:
        print( "❎ Weighted Score Not Applied (AI Score Used)\n")

    result_str += f"🔹 Weighted Score for real video: {weighted_score:.6f}\n"
    final_prediction = "REAL" if weighted_score >= 0.5 else "FAKE"
    result_str += f"✅ Final Prediction: {final_prediction}\n"
    result_str += "--------------------------------------------------\n"

    print(result_str)
    return { #필요한 경우 여기서 return한 후에 프론트에 연결하면 됨
        "video_id": video_id,
        "ai_score_real": score,
        "ai_score_fake": fscore,
        "weighted_score_real": weighted_score,
        "user_votes_considered": user_count if 0.3 <= score <= 0.6 else None,
        "user_vote_ratio": p_u if 0.3 <= score <= 0.6 else None,
        "final_prediction": final_prediction,
        "result_text": result_str
    }

# ✅ Execution Logic
'''
if __name__ == "__main__":
    video_id = input("🎥 Enter Video ID: ")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = download_youtube_video(video_url)
    #여기서 video_id만 바꾸면 될 듯
    print(f"🔍 Analysis Started: {video_path}")
    predict_and_print_results(video_id, video_path)
'''