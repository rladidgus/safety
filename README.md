<h1>미니프로젝트 safety-chat 실행하기</h1>

<h2>1. 실행 프로그램 설치</h2>
VScode 또는 Antigravity

<h2>2. 파일 경로 지정 및 불러오기</h2>
<h3>[1] C드라이브 내부에 파일을 생성하고 불러온다.</h3>
	Ex.) user 계정 안에 safety-chat 폴더를 넣고 불러온다.

<h3>[2] Antigravity에서 safety-chat 열기</h3>
	File > openFolder > safety-chat 클릭 > [폴더 선택]

<h2>3. 가상환경 설정</h2>
<h3>[1] conda-forge 설치하기</h3>
https://conda-forge.org/ > Download Installer 클릭 후 설치


<h3>[2] 내부 터미널 열기 (command Prompt 사용 권장)</h3>
다음과 같이 입력한다.
C:\Users\..[폴더경로]> conda create -n [가상환경 명] python=[파이썬 버전]

Ex.)
C:\Users\Admin\Working> conda create -n Proj python=3.11

다음과 같이 표시되면 가상환경 접속 완료
(Proj) C:\Users\Admin\safety-chat>

<h2>4. 환경 실행하기</h2>
<h3>[1] 백엔드 환경 실행</h3>
	<h5>(1)backend 폴더 경로 접근</h5>
	(Proj) C:\Users\Admin\safety-chat> 상태에서 cd backend 입력

	<h5>(2) uvicorn 실행</h5>
	(Proj) C:\Users\Admin\safety-chat\backend>uvicorn app.main:app --reload
	
	<h4>[라이브러리 설치]<h4>
		pip install -r requirements.txt 입력한다. (osmnx는 포함되지 않음.)
	이때 'uvicorn', 'osmnx' 라이브러리를 설치해야 한다.
  ----------------------------------------
		uvicorn 설치 > pip install uvicorn (requirements.txt에 포함되어)
    --------------------------------------
		osmnx 설치 > pip install osmnx
		설치 확인 : python -c "import osmnx; print('osmnx installed successfully')"을 입력하여 
				'osmnx installed successfully' 문구가 표시되면 설치 완료
		bcrypt 라이브러리 설치 > pip install bcrypt
<h3>[2] 프론트엔드 환경 실행</h3>
<h5>(1) 가상 환경 상태에서 cd frontend 입력</h5>
(Proj) C:\Users\Admin\safety-chat> cd frontend

<h5>(2) 리엑트 실행</h5>
(MiniProj) C:\Users\Admin\safety-chat\frontend>npm run dev

<h2>[최종 실행]</h2>
http://localhost:5173/
