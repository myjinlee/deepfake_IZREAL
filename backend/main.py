from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import download_youtube_video
from model import predict_video

from fastapi import Depends
from sqlalchemy.orm import Session
from model import SessionLocal, VideoVote

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

# ✅ 딥페이크 예측 엔드포인트
@app.post("/predict/")
async def predict_video_endpoint(payload: dict):
    url = payload.get("url")
    try:
        # 유튜브 영상 다운로드
        video_path = download_youtube_video(url)

        # 딥페이크 예측
        result = predict_video(video_path)

        return JSONResponse(content={
            "message": result["message"], 
            "real_score": result["real_score"],
            "fake_score": result["fake_score"]
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


# ! 추후 투표 결과를 분석하기 위해 get 엔드포인트 ㅜㅊ가
@app.get("/votes/{video_id}")
async def get_votes(video_id: str, db: Session = Depends(get_db)):
    votes = db.query(VideoVote).filter(VideoVote.video_id == video_id).all()
    
    total_votes = len(votes)
    deepfake_votes = sum(1 for v in votes if v.vote)
    real_votes = total_votes - deepfake_votes

    return {
        "total_votes": total_votes,
        "deepfake_votes": deepfake_votes,
        "real_votes": real_votes
    }


