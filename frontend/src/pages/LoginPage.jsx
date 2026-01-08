import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const LoginPage = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();

    const handleLogin = async () => {
        try {
            // For now, let's keep the API call as requested in the source HTML
            // But if the backend isn't ready, we might want a fallback or just alert.
            // As per instructions, I'll implement the fetch.
            // If the user hasn't set up the backend proxy or API, this might fail,
            // so I will add a fallback for demonstration purposes if valid credentials are mostly testing.

            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const result = await response.json();

            if (response.ok) {
                alert("로그인 성공!");
                navigate('/main');
            } else {
                alert("로그인 실패! : " + (result.detail || "Unknown error"));
            }
        } catch (error) {
            console.error("Login error:", error);
            // Fallback for demo if no backend:
            // alert("서버 연결 실패. (데모 모드로 넘어갑니다)");
            // navigate('/main');

            alert("서버 연결에 실패했습니다.");
        }
    };

    return (
        <div className="login-container-wrapper">
            <div className="login-container">
                <h2>로그인</h2>
                <input
                    type="text"
                    placeholder="아이디"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                />
                <input
                    type="password"
                    placeholder="비밀번호"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
                <button onClick={handleLogin}>로그인</button>
                <div className="link-text">
                    계정이 없으신가요? <a href="/signup">회원가입</a>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
