// src/pages/ResultPage.jsx
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import MapContainer from '../components/MapContainer';

const ResultPage = () => {
    const location = useLocation();
    const navigate = useNavigate();

    // MAIN INPUT: Address strings from MainPage
    const { start, end, type } = location.state || {};

    const [routeData, setRouteData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');

    useEffect(() => {
        if (!start || !end) {
            navigate('/main', { replace: true });
            return;
        }

        const fetchRoute = async () => {
            try {
                setLoading(true);
                setErrorMsg('');

                // We delegate Geocoding to the Backend to avoid Client-side SDK issues
                const payload = {
                    start_address: start,
                    end_address: end,
                    mode: type || 'safe'
                };

                console.log("Sending payload to backend (no coords):", payload);

                const res = await fetch('http://localhost:8000/api/route/safe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                if (!res.ok) {
                    const text = await res.text();
                    console.error('route api error:', res.status, text);
                    setErrorMsg('경로를 찾을 수 없습니다. (주소를 확인해주세요)');
                    return;
                }

                const data = await res.json();
                setRouteData(data);

            } catch (e) {
                console.error('fetch error:', e);
                setErrorMsg(`서버 연결 오류: ${e.message}`);
            } finally {
                setLoading(false);
            }
        };

        fetchRoute();
    }, [start, end, type, navigate]);

    return (
        <div className="container">
            <div className="card result-card">
                <div className="header-simple">
                    <button className="back-button" onClick={() => navigate(-1)}>←</button>
                    <h2 className="title">안전 경로 안내</h2>
                </div>

                <div className="map-area-result">
                    {routeData ? (
                        <MapContainer routeData={routeData} />
                    ) : loading ? (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '10px' }}>
                            <div className="spinner" style={{ width: 30, height: 30, border: '4px solid #f3f3f3', borderTop: '4px solid #3498db', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                            {/* Simple inline spinner style */}
                            <style>{`@keyframes spin {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}`}</style>
                            <div>경로를 계산하고 있습니다...</div>
                            <div style={{ fontSize: '0.8rem', color: '#666' }}>안전 데이터를 분석 중입니다.</div>
                        </div>
                    ) : errorMsg ? (
                        <div style={{ padding: 20, color: 'red', textAlign: 'center' }}>
                            {errorMsg}
                            <br />
                            <button className="btn btn-outline small" style={{ marginTop: 10 }} onClick={() => navigate(-1)}>뒤로 가기</button>
                        </div>
                    ) : (
                        <div style={{ padding: 20 }}>데이터 없음</div>
                    )}
                </div>

                <div className="info-card">
                    {routeData ? (
                        <>
                            <div className="info-row">
                                <span>⏱️ {(routeData.duration / 60).toFixed(0)}분</span>
                                <span style={{ marginLeft: 8 }}>·</span>
                                <span style={{ marginLeft: 8 }}>
                                    🚶 {(routeData.distance / 1000).toFixed(1)}km
                                </span>
                            </div>
                            <div className="info-row">
                                <strong>안전도 점수:</strong> <span style={{ color: '#2563eb', marginLeft: 5 }}>{routeData.safety_score.toFixed(1)}점</span>
                            </div>
                        </>
                    ) : (
                        <div className="info-row">
                            ⏱️ --분 · 🚶 --km
                        </div>
                    )}

                    <div className="info-highlight">
                        <span className="safe-badge">
                            {type === 'shortest' ? '최단 경로' : '안전 우선 경로'}
                        </span>
                    </div>
                </div>

                <div className="bottom-actions">
                    <button className="btn btn-outline small" onClick={() => navigate(-1)}>다시 검색</button>
                    <button className="btn btn-primary small">길안내 시작</button>
                </div>
            </div>
        </div>
    );
};

export default ResultPage;
