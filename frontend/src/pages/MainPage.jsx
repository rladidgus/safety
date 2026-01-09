// src/pages/MainPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../index.css';

const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search';

// 프리셋 경로
const PRESETS = [
  { name: '서울역 → 시청', start: { name: '서울역', lat: 37.5546, lon: 126.9706 }, end: { name: '시청', lat: 37.5665, lon: 126.9780 } },
  { name: '강남역 → 삼성역', start: { name: '강남역', lat: 37.4979, lon: 127.0276 }, end: { name: '삼성역', lat: 37.5089, lon: 127.0631 } },
  { name: '홍대 → 신촌', start: { name: '홍대입구', lat: 37.5563, lon: 126.9237 }, end: { name: '신촌', lat: 37.5597, lon: 126.9427 } },
];

const MainPage = () => {
  const navigate = useNavigate();
  const [startPoint, setStartPoint] = useState('');
  const [endPoint, setEndPoint] = useState('');
  const [startCoord, setStartCoord] = useState(null);
  const [endCoord, setEndCoord] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // 주소 → 좌표 변환 (Geocoding)
  const geocode = async (address) => {
    const response = await fetch(
      `${NOMINATIM_URL}?format=json&q=${encodeURIComponent(address + ', Seoul, South Korea')}&limit=1`,
      { headers: { 'Accept-Language': 'ko' } }
    );
    const results = await response.json();
    if (results.length > 0) {
      return { lat: parseFloat(results[0].lat), lon: parseFloat(results[0].lon) };
    }
    return null;
  };

  // 경로 비교 검색
  const handleSearch = async () => {
    if (!startPoint.trim() || !endPoint.trim()) {
      setError('출발지와 목적지를 입력해주세요.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      let start = startCoord;
      let end = endCoord;

      // 좌표가 없으면 geocoding
      if (!start) {
        start = await geocode(startPoint);
        if (!start) {
          setError('출발지를 찾을 수 없습니다.');
          setLoading(false);
          return;
        }
        setStartCoord(start);
      }

      if (!end) {
        end = await geocode(endPoint);
        if (!end) {
          setError('목적지를 찾을 수 없습니다.');
          setLoading(false);
          return;
        }
        setEndCoord(end);
      }

      // ResultPage로 이동 (좌표 포함)
      navigate('/result', {
        state: {
          start: startPoint,
          end: endPoint,
          startCoord: start,
          endCoord: end
        }
      });
    } catch (e) {
      console.error('Geocoding error:', e);
      setError('주소 검색 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 프리셋 선택
  const handlePreset = (preset) => {
    setStartPoint(preset.start.name);
    setEndPoint(preset.end.name);
    setStartCoord({ lat: preset.start.lat, lon: preset.start.lon });
    setEndCoord({ lat: preset.end.lat, lon: preset.end.lon });

    // 바로 이동
    navigate('/result', {
      state: {
        start: preset.start.name,
        end: preset.end.name,
        startCoord: { lat: preset.start.lat, lon: preset.start.lon },
        endCoord: { lat: preset.end.lat, lon: preset.end.lon }
      }
    });
  };

  return (
    <div className="container">
      <div className="card">
        <div className="header">
          <h1 className="service-title">🛡️ 안전 경로</h1>
          <button className="menu-button">☰</button>
        </div>

        <div className="input-group">
          <div className="input-wrapper">
            <span className="icon">📍</span>
            <input
              type="text"
              placeholder="출발지를 입력하세요"
              value={startPoint}
              onChange={(e) => {
                setStartPoint(e.target.value);
                setStartCoord(null); // 좌표 초기화
              }}
            />
          </div>
          <div className="input-wrapper">
            <span className="icon">🏁</span>
            <input
              type="text"
              placeholder="목적지를 입력하세요"
              value={endPoint}
              onChange={(e) => {
                setEndPoint(e.target.value);
                setEndCoord(null); // 좌표 초기화
              }}
            />
          </div>
        </div>

        {error && (
          <div style={{ color: 'red', textAlign: 'center', margin: '10px 0', fontSize: '14px' }}>
            ⚠️ {error}
          </div>
        )}

        <div className="button-group">
          <button
            className="btn btn-primary"
            onClick={handleSearch}
            disabled={loading}
            style={{ width: '100%' }}
          >
            {loading ? '검색 중...' : '🔍 경로 비교 검색'}
          </button>
        </div>

        {/* 프리셋 버튼 */}
        <div style={{ marginTop: '15px' }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', textAlign: 'center' }}>
            빠른 테스트
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
            {PRESETS.map((preset, idx) => (
              <button
                key={idx}
                className="btn btn-outline small"
                onClick={() => handlePreset(preset)}
                style={{ fontSize: '12px', padding: '6px 12px' }}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        <div className="map-placeholder-main" style={{ marginTop: '20px' }}>
          <div className="map-area-text">
            <span>📍 지도는 결과 페이지에서 표시됩니다</span>
          </div>
        </div>

        <div className="bottom-link">
          <a href="#">안전 경로란?</a>
        </div>
      </div>
    </div>
  );
};

export default MainPage;
