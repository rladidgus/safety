import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const SignupPage = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();

    const handleSignup = async () => {
        try {
            const response = await fetch('/api/signup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const result = await response.json();

            if (response.ok) {
                alert("회원가입 성공! 로그인 페이지로 이동합니다.");
                navigate('/');
            } else {
                alert("오류: " + (result.detail || "Unknown error"));
            }
        } catch (error) {
            console.error("Signup error:", error);
            // Fallback demo for consistency if backend is missing
            alert("서버 연결에 실패했습니다.");
        }
    };

    return (
        <div className="login-container-wrapper">
            <div className="login-container">
                <h2>회원가입</h2>
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
                <button className="btn-success" onClick={handleSignup}>가입하기</button>
                <div className="link-text">
                    이미 계정이 있으신가요? <a href="/">로그인</a>
                </div>
            </div>
        </div>
    );
};

export default SignupPage;
