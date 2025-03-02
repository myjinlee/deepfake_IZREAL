from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import download_youtube_video
from model import predict_video

from urllib.parse import urlparse, parse_qs
from fastapi import Depends
from sqlalchemy.orm import Session
from model import SessionLocal, VideoVote

# ! newmodel.py에서 필요한 함수 import
from newmodel import predict_and_print_results

app = FastAPI()

# ✅ 데이터베이스 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 경로 (GET 요청 처리)
@app.get("/")
async def root():
    return {"message": "Welcome to the Deepfake Detection API!"}

#!이거 추가
from urllib.parse import urlparse, parse_qs

def getYouTubeVideoId(url):
    try:
        parsed_url = urlparse(url)

        # ✅ youtu.be 단축 URL 처리 (예: https://youtu.be/oPbuyJqSQ2k)
        if parsed_url.hostname == "youtu.be":
            return parsed_url.path[1:]

        # ✅ YouTube Shorts URL 처리 (예: https://www.youtube.com/shorts/oPbuyJqSQ2k)
        elif "/shorts/" in parsed_url.path:
            return parsed_url.path.split("/shorts/")[1]  # "/shorts/" 뒤의 값이 video ID

        # ✅ 일반적인 YouTube URL 처리 (예: https://www.youtube.com/watch?v=oPbuyJqSQ2k)
        elif parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]

        return None  # 올바르지 않은 URL이면 None 반환
    except Exception:
        return None


#✅ 딥페이크 예측 엔드포인트
@app.post("/predict/")
async def predict_video_endpoint(payload: dict):
    url = payload.get("url")
    video_id = getYouTubeVideoId(url)

    try:
        # 유튜브 영상 다운로드
        video_path = download_youtube_video(url)
        
        # 딥페이크 예측 1
        print(f"🔍 Analysis Started: {video_path}")
        result1 = predict_and_print_results(video_id, video_path)
    
        # 예측 2
        
        result = predict_video(video_path)
        
        return JSONResponse(content={
            "message": result["message"], 
            "real_score": result["real_score"],
            "fake_score": result["fake_score"],
            "result_text": result1["result_text"]
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ 투표 API 추가
class VoteRequest(BaseModel):
    video_id: str
    user_id: str
    vote: bool

votes = {}

@app.post("/vote/")
async def vote(request: VoteRequest, db: Session = Depends(get_db)):
    if not request.video_id or not request.user_id:
        return JSONResponse(content={"error": "Missing video_id or user_id"}, status_code=400)

    # ✅ MySQL에 투표 데이터 저장
    new_vote = VideoVote(video_id=request.video_id, user_id=request.user_id, vote=request.vote)
    db.add(new_vote)
    db.commit()

    return JSONResponse(content={"message": f"투표 완료: {'딥페이크' if request.vote else '진짜'}로 선택됨"})

