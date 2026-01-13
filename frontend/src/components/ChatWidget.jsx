import React, { useState, useRef, useEffect } from 'react';
import './ChatWidget.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

const SUGGESTIONS = [
    '위험 지역 알려줘',
    '주변 안전 시설 알려줘',

];

function ChatWidget({ currentLat = 37.5665, currentLng = 126.9780, onMoveTo, onDrawRoute }) {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { text: '안녕하세요! 현재 위치의 안전 정보가 궁금하시면 물어봐주세요.', sender: 'bot' }
    ]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const sendMessage = async (messageText) => {
        const text = messageText || inputValue.trim();
        if (!text) return;

        // Add user message
        setMessages(prev => [...prev, { text, sender: 'user' }]);
        setInputValue('');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    current_lat: currentLat,
                    current_lng: currentLng
                })
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const data = await response.json();

            // Add bot response
            setMessages(prev => [...prev, { text: data.reply, sender: 'bot' }]);

            // Handle map actions
            if (data.route_data && onDrawRoute) {
                onDrawRoute(data.route_data);
            }
            if (data.move_to && onMoveTo) {
                onMoveTo(data.move_to.lat, data.move_to.lng);
            }

        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, {
                text: '오류가 발생했습니다. 다시 시도해주세요.',
                sender: 'bot'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    };

    return (
        <>
            {/* Floating Button */}
            <button
                className="chat-fab"
                onClick={() => setIsOpen(!isOpen)}
                aria-label="채팅 열기"
            >
                💬
            </button>

            {/* Chat Window */}
            {isOpen && (
                <div className="chat-window">
                    <div className="chat-header">
                        <span>안심 길 안내 도우미</span>
                        <span className="chat-close" onClick={() => setIsOpen(false)}>×</span>
                    </div>

                    <div className="chat-messages">
                        {messages.map((msg, index) => (
                            <div
                                key={index}
                                className={`chat-message ${msg.sender}`}
                            >
                                {msg.text}
                            </div>
                        ))}

                        {isLoading && (
                            <div className="typing-indicator">
                                <span className="typing-dot"></span>
                                <span className="typing-dot"></span>
                                <span className="typing-dot"></span>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>

                    <div className="chat-suggestions">
                        {SUGGESTIONS.map((suggestion, index) => (
                            <div
                                key={index}
                                className="suggestion-chip"
                                onClick={() => sendMessage(suggestion)}
                            >
                                {suggestion}
                            </div>
                        ))}
                    </div>

                    <div className="chat-input-area">
                        <input
                            type="text"
                            className="chat-input"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="메시지를 입력하세요..."
                        />
                        <button
                            className="chat-send-btn"
                            onClick={() => sendMessage()}
                        >
                            전송
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}

export default ChatWidget;
