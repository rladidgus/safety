from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel
import models, database

app = FastAPI()
models.Base.metadata.create_all(bind=database.engine)
templates = Jinja2Templates(directory="templates")

# DB 세션 의존성
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserData(BaseModel):
    username: str
    password: str

# 초기 화면을 로그인 페이지로 설정
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# 회원가입 페이지 경로 추가
@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/api/signup")
async def signup(data: UserData, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
    
    new_user = models.User(username=data.username, password=data.password)
    db.add(new_user)
    db.commit()
    return {"message": "signup success"}

@app.post("/api/login")
async def login(data: UserData, response: Response, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        models.User.username == data.username,
        models.User.password == data.password
        ).first()
        
    if not user:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    response.set_cookie(key="username", value=user.username)
    return {"message": "success"}

# 2. 로그아웃 API 추가: 쿠키 삭제
@app.post("/api/logout")
async def logout(response: Response):
    response.delete_cookie("username")
    return {"message": "logout success"}


@app.get("/main", response_class=HTMLResponse)
async def main_page(request: Request):
    # 쿠키에서 username 읽기
    username = request.cookies.get("username")
    is_logged_in = True if username else False

    return templates.TemplateResponse("main.html", {
        "request": request, 
        "is_logged_in": is_logged_in,
        "username": username
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)