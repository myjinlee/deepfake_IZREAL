import React, { useState } from "react";

const VotePopup = ({ videoId, onClose }) => {
    const [vote, setVote] = useState(null);
    const [message, setMessage] = useState("");

    const handleVote = async (isDeepfake) => {
        setVote(isDeepfake);
        try {
            const response = await fetch("http://localhost:8000/vote/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_id: videoId,
                    user_id: "test_user", // 실제 사용자 ID가 있으면 넣기
                    vote: isDeepfake
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                setMessage(`투표 완료: ${isDeepfake ? "딥페이크" : "진짜"}로 선택됨`);
            } else {
                setMessage(`투표 실패: ${data.message || "서버 오류 발생"}`);
            }
        } catch (error) {
            setMessage("서버 연결 오류");
        }

        // 투표 완료 후 1초 후 팝업 닫기
        setTimeout(() => {
            onClose();
        }, 1000);
    };

    return (
        <div className="popup">
            <h2>이 영상이 딥페이크라고 생각하시나요?</h2>
            <button onClick={() => handleVote(true)}>Yes</button>
            <button onClick={() => handleVote(false)}>No</button>
            <p>{message}</p>
        </div>
    );
};

export default VotePopup;