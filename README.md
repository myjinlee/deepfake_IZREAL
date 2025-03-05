첨부할 이미지
  1. 구현 페이지 (시작 페이지, 클릭한 후 투표페이지, 수치 바뀐 마지막 페이지 첨부 - 보고서에도 작성)

리드미 작성 방식 
  로컬 가상환경 설정 방식
  backend, frontend 설정 방식





# 실습 환경 구성

backend, frontend, MySQL (DB) 환경 설정을 위한 설명. 

## backend 환경 설정
1. 새 cmd open 
2. 프로젝트 폴더에 가상환경 만들고 activate  :
     python -m venv testvenv => venv\Scripts\activate
4. cd backend 로 backend 페이지로 이동한 후 FastAPI 실행을 위한 uvicorn main:app --reload 입력


## MySQL(DB) 환경 설정
1. 로컬에 MySQL 이 설치되어 있어야 함.
2. 새 cmd open
3. mysql -u root -p 입력 후 엔터 => Enter password : 에 본인의 MySQL 비밀번호 입력
4. video_db 이름의 DB 만들기 -> CREATE DATABASE video_db { 어쩌구 } 스크린 캡쳐해서 붙이기 
5. USE video_db; 입력하고 cmd 에 DATABASE changed 출력 확인
   


## frontend 환경설정
1. 로컬에 node.js 가 설치되어 있어야 함. 
2. 새 cmd open
3. 프로젝트 폴더의 frontend 폴더로 이동 (cd frontend 입력)
4. cmd에 npm <install ~ 설치문구 입력 예정 ~ .js> 입력
5. npm start 입력후 localhost:3000 부분 Ctrl+click 으로 접속
