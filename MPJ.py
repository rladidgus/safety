from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# HTML 템플릿 설정을 위해 templates 폴더 지정
templates = Jinja2Templates(directory="templates")

# 로그인 데이터를 받기 위한 Pydantic 모델 정의
class LoginRequest(BaseModel):
    username: str
    password: str

# 1. 초기 화면 (로그인/회원가입 페이지) 보여주기
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. 로그인 요청 처리 (POST 방식)
@app.post("/api/login")
async def login(data: LoginRequest):
    # 실제 DB 연동 전 임시 테스트용 계정
    if data.username == "admin" and data.password == "1234":
        return {"message": "success"}
    else:
        # 실패 시 401 Unauthorized 에러 반환
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="fail"
        )

# 3. 로그인 성공 후 이동할 메인 페이지
@app.get("/main", response_class=HTMLResponse)
async def main(request: Request):
    # Flask 코드에서 문자열만 있던 부분은 실제 로직에 포함되지 않으므로 템플릿만 반환합니다.
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    # Flask의 app.run과 유사한 역할 (5000번 포트 실행)
    uvicorn.run(app, host="0.0.0.0", port=8000)