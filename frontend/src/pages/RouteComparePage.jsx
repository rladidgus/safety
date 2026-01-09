// src/pages/RouteComparePage.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import useRouteCompare from '../hooks/useRouteCompare';
import './RouteComparePage.css';

// 프리셋 경로
const PRESETS = {
    'seoul-city': {
        name: '서울역 → 시청',
        start: { name: '서울역', lat: 37.5546, lon: 126.9706 },
        end: { name: '시청', lat: 37.5665, lon: 126.9780 }
    },
    'gangnam': {
        name: '강남역 → 삼성역',
        start: { name: '강남역', lat: 37.4979, lon: 127.0276 },
        end: { name: '삼성역', lat: 37.5089, lon: 127.0631 }
    },
    'jongno': {
        name: '종로 → 동대문',
        start: { name: '종로', lat: 37.5700, lon: 126.9830 },
        end: { name: '동대문', lat: 37.5711, lon: 127.0095 }
    },
    'hongdae': {
        name: '홍대 → 신촌',
        start: { name: '홍대입구', lat: 37.5563, lon: 126.9237 },
        end: { name: '신촌', lat: 37.5597, lon: 126.9427 }
    }
};

const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search';

function RouteComparePage() {
    const mapRef = useRef(null);
    const mapInstanceRef = useRef(null);
    const markersRef = useRef([]);
    const polylinesRef = useRef([]);

    const [startCoord, setStartCoord] = useState(null);
    const [endCoord, setEndCoord] = useState(null);
    const [startName, setStartName] = useState('');
    const [endName, setEndName] = useState('');
    const [startResults, setStartResults] = useState([]);
    const [endResults, setEndResults] = useState([]);
    const [showStartResults, setShowStartResults] = useState(false);
    const [showEndResults, setShowEndResults] = useState(false);

    const { loading, error, result, compareRoutes, reset } = useRouteCompare();

    // 지도 초기화
    useEffect(() => {
        if (!mapRef.current || mapInstanceRef.current) return;

        const map = L.map(mapRef.current).setView([37.5665, 126.9780], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        // WMS 레이어 (범죄주의구간)
        L.tileLayer.wms("https://www.safemap.go.kr/openapi2/IF_0087_WMS", {
            layers: "A2SM_CRMNLHSPOT_TOT",
            styles: "A2SM_CrmnlHspot_Tot_Tot",
            format: "image/png",
            transparent: true,
            opacity: 0.4,
            serviceKey: "7F2ABSQ6-7F2A-7F2A-7F2A-7F2ABSQ61U"
        }).addTo(map);

        mapInstanceRef.current = map;

        return () => {
            map.remove();
            mapInstanceRef.current = null;
        };
    }, []);

    // 마커 클리어
    const clearMarkers = useCallback(() => {
        markersRef.current.forEach(m => m.remove());
        markersRef.current = [];
    }, []);

    // 폴리라인 클리어
    const clearPolylines = useCallback(() => {
        polylinesRef.current.forEach(p => p.remove());
        polylinesRef.current = [];
    }, []);

    // 마커 추가
    const addMarker = useCallback((lat, lon, text, color) => {
        if (!mapInstanceRef.current) return;

        const icon = L.divIcon({
            className: 'custom-marker-icon',
            html: `<div style="background:${color};color:white;padding:5px 10px;border-radius:20px;font-weight:bold;font-size:12px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.3);transform:translate(-50%,-100%);">${text}</div>`,
            iconSize: null,
            iconAnchor: [0, 0]
        });

        const marker = L.marker([lat, lon], { icon }).addTo(mapInstanceRef.current);
        markersRef.current.push(marker);
    }, []);

    // 프리셋 선택
    const handlePreset = (key) => {
        const preset = PRESETS[key];
        setStartName(preset.start.name);
        setEndName(preset.end.name);
        setStartCoord({ lat: preset.start.lat, lon: preset.start.lon });
        setEndCoord({ lat: preset.end.lat, lon: preset.end.lon });

        clearMarkers();
        addMarker(preset.start.lat, preset.start.lon, '출발', '#22c55e');
        addMarker(preset.end.lat, preset.end.lon, '도착', '#ef4444');

        if (mapInstanceRef.current) {
            mapInstanceRef.current.fitBounds([
                [preset.start.lat, preset.start.lon],
                [preset.end.lat, preset.end.lon]
            ], { padding: [50, 50] });
        }
    };

    // 위치 검색
    const searchLocation = async (query, isStart) => {
        if (query.length < 2) {
            isStart ? setShowStartResults(false) : setShowEndResults(false);
            return;
        }

        try {
            const response = await fetch(
                `${NOMINATIM_URL}?format=json&q=${encodeURIComponent(query + ', Seoul, South Korea')}&limit=5`,
                { headers: { 'Accept-Language': 'ko' } }
            );
            const results = await response.json();

            if (isStart) {
                setStartResults(results);
                setShowStartResults(true);
            } else {
                setEndResults(results);
                setShowEndResults(true);
            }
        } catch (e) {
            console.error('검색 오류:', e);
        }
    };

    // 위치 선택
    const selectLocation = (item, isStart) => {
        const lat = parseFloat(item.lat);
        const lon = parseFloat(item.lon);
        const name = item.display_name.split(',')[0];

        if (isStart) {
            setStartName(name);
            setStartCoord({ lat, lon });
            setShowStartResults(false);
        } else {
            setEndName(name);
            setEndCoord({ lat, lon });
            setShowEndResults(false);
        }

        clearMarkers();
        if (isStart) {
            addMarker(lat, lon, '출발', '#22c55e');
            if (endCoord) addMarker(endCoord.lat, endCoord.lon, '도착', '#ef4444');
        } else {
            if (startCoord) addMarker(startCoord.lat, startCoord.lon, '출발', '#22c55e');
            addMarker(lat, lon, '도착', '#ef4444');
        }
    };

    // 경로 검색
    const handleSearch = async () => {
        if (!startCoord || !endCoord) {
            alert('출발지와 목적지를 선택해주세요.');
            return;
        }

        clearPolylines();
        reset();

        try {
            const data = await compareRoutes(
                startCoord.lat, startCoord.lon,
                endCoord.lat, endCoord.lon
            );

            // 최단 경로 (파란색, 점선) - 먼저 그려서 아래에 위치
            const shortestLatLngs = data.shortest.path_coords.map(c => [c[0], c[1]]);
            const shortestLine = L.polyline(shortestLatLngs, {
                color: '#3b82f6',
                weight: 5,
                opacity: 0.9,
                dashArray: '10, 10'  // 점선
            }).addTo(mapInstanceRef.current);
            polylinesRef.current.push(shortestLine);

            // 안전 경로 (초록색, 실선) - 나중에 그려서 위에 위치
            const safestLatLngs = data.safest.path_coords.map(c => [c[0], c[1]]);
            const safestLine = L.polyline(safestLatLngs, {
                color: '#22c55e',
                weight: 6,
                opacity: 0.9
            }).addTo(mapInstanceRef.current);
            polylinesRef.current.push(safestLine);

            // 둘 다 보이게 범위 조정
            const allCoords = [...shortestLatLngs, ...safestLatLngs];
            mapInstanceRef.current.fitBounds(allCoords, { padding: [50, 50] });

        } catch (e) {
            console.error('경로 검색 실패:', e);
        }
    };

    // 현재 시간 정보
    const getTimeInfo = () => {
        const now = new Date();
        const hour = now.getHours();
        const streetlightOn = hour >= 18 || hour < 6;
        return `${hour.toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')} (가로등: ${streetlightOn ? 'ON 🌙' : 'OFF ☀️'})`;
    };

    return (
        <div className="compare-container">
            <header className="compare-header">
                <h1>🛡️ 경로 비교</h1>
                <p>최단 경로 vs 안전 경로</p>
                <div className="time-info">⏰ 현재 시간: {getTimeInfo()}</div>
            </header>

            <div className="compare-main">
                <div className="compare-sidebar">
                    <div className="form-group">
                        <label>📍 출발지</label>
                        <input
                            type="text"
                            value={startName}
                            onChange={(e) => {
                                setStartName(e.target.value);
                                searchLocation(e.target.value, true);
                            }}
                            placeholder="출발지 검색"
                        />
                        {showStartResults && startResults.length > 0 && (
                            <div className="search-results">
                                {startResults.map((r, i) => (
                                    <div
                                        key={i}
                                        className="search-result-item"
                                        onClick={() => selectLocation(r, true)}
                                    >
                                        {r.display_name.split(',').slice(0, 2).join(', ')}
                                    </div>
                                ))}
                            </div>
                        )}
                        {startCoord && (
                            <div className="coord-display">
                                좌표: {startCoord.lat.toFixed(4)}, {startCoord.lon.toFixed(4)}
                            </div>
                        )}
                    </div>

                    <div className="form-group">
                        <label>🏁 목적지</label>
                        <input
                            type="text"
                            value={endName}
                            onChange={(e) => {
                                setEndName(e.target.value);
                                searchLocation(e.target.value, false);
                            }}
                            placeholder="목적지 검색"
                        />
                        {showEndResults && endResults.length > 0 && (
                            <div className="search-results">
                                {endResults.map((r, i) => (
                                    <div
                                        key={i}
                                        className="search-result-item"
                                        onClick={() => selectLocation(r, false)}
                                    >
                                        {r.display_name.split(',').slice(0, 2).join(', ')}
                                    </div>
                                ))}
                            </div>
                        )}
                        {endCoord && (
                            <div className="coord-display">
                                좌표: {endCoord.lat.toFixed(4)}, {endCoord.lon.toFixed(4)}
                            </div>
                        )}
                    </div>

                    <div className="preset-buttons">
                        {Object.entries(PRESETS).map(([key, preset]) => (
                            <button
                                key={key}
                                className="btn-preset"
                                onClick={() => handlePreset(key)}
                            >
                                {preset.name}
                            </button>
                        ))}
                    </div>

                    <button
                        className="btn-primary"
                        onClick={handleSearch}
                        disabled={loading}
                    >
                        {loading ? '검색 중...' : '🔍 경로 검색'}
                    </button>

                    {error && <div className="error-box">⚠️ {error}</div>}

                    {result && (
                        <div className="results">
                            <div className="result-card shortest">
                                <h4>🔵 최단 경로</h4>
                                <div className="result-stats">
                                    <div className="stat">
                                        <div className="stat-value">{result.shortest.length.toFixed(0)}m</div>
                                        <div className="stat-label">거리</div>
                                    </div>
                                    <div className="stat">
                                        <div className="stat-value">{result.shortest.safety_score}점</div>
                                        <div className="stat-label">안전점수</div>
                                    </div>
                                </div>
                            </div>

                            <div className="result-card safest">
                                <h4>🟢 안전 경로</h4>
                                <div className="result-stats">
                                    <div className="stat">
                                        <div className="stat-value">{result.safest.length.toFixed(0)}m</div>
                                        <div className="stat-label">거리</div>
                                    </div>
                                    <div className="stat">
                                        <div className="stat-value">{result.safest.safety_score}점</div>
                                        <div className="stat-label">안전점수</div>
                                    </div>
                                </div>
                            </div>

                            <div className="comparison">
                                <h4>📊 비교</h4>
                                <p>
                                    <span className="diff-length">+{result.length_difference.toFixed(0)}m</span> 더 걸어서{' '}
                                    <span className="diff-safety">+{result.safety_improvement}점</span> 더 안전
                                </p>
                            </div>
                        </div>
                    )}
                </div>

                <div className="compare-map-container">
                    <div ref={mapRef} className="compare-map" />
                    <div className="map-legend">
                        <div className="legend-item">
                            <span className="legend-line shortest"></span>
                            <span>최단 경로</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-line safest"></span>
                            <span>안전 경로</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default RouteComparePage;
