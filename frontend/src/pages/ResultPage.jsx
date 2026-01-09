// src/pages/ResultPage.jsx
import React, { useEffect, useState, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import 'leaflet.markercluster';
import ChatWidget from '../components/ChatWidget';

// --- Custom Icons ---
const cctvIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41]
});
const streetIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-gold.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41]
});
const entIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41]
});
const policeIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-violet.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41]
});
const convIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41]
});

// --- Cluster Styles ---
const clusterStyles = `
    .custom-cluster-icon {
        background-clip: padding-box;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        line-height: 40px !important;
        color: white;
        border: 2px solid white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .marker-cluster-cctv { background-color: rgba(59, 130, 246, 0.66); }
    .marker-cluster-street { background-color: rgba(234, 179, 8, 0.66); }
    .marker-cluster-ent { background-color: rgba(239, 68, 68, 0.66); }
    .marker-cluster-poli { background-color: rgba(255, 255, 255, 0.66); }
    .marker-cluster-conv { background-color: rgba(52, 199, 89, 0.66); }
    .custom-cluster-icon div {
        width: 36px; height: 36px; margin: 0;
        text-align: center; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px;
    }
`;

const ResultPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const mapRef = useRef(null);
    const mapInstanceRef = useRef(null);

    const { start, end, startCoord, endCoord } = location.state || {};

    const [compareData, setCompareData] = useState(null);
    const [points, setPoints] = useState([]);
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');
    const [activeRoute, setActiveRoute] = useState('both');

    // 포인트 데이터 가져오기
    useEffect(() => {
        fetch('http://localhost:8000/api/points')
            .then(res => res.json())
            .then(data => {
                if (data.points) setPoints(data.points);
            })
            .catch(err => console.error("Error fetching points:", err));
    }, []);

    // 경로 비교 API 호출
    useEffect(() => {
        if (!startCoord || !endCoord) {
            navigate('/main', { replace: true });
            return;
        }

        const fetchCompare = async () => {
            try {
                setLoading(true);
                setErrorMsg('');

                const res = await fetch('http://localhost:8000/api/route/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        start_lat: startCoord.lat,
                        start_lon: startCoord.lon,
                        end_lat: endCoord.lat,
                        end_lon: endCoord.lon
                    }),
                });

                if (!res.ok) {
                    const errData = await res.json();
                    setErrorMsg(errData.detail || '경로를 찾을 수 없습니다.');
                    return;
                }

                const data = await res.json();
                console.log('Compare API response:', data);
                setCompareData(data);

            } catch (e) {
                console.error('fetch error:', e);
                setErrorMsg(`서버 연결 오류: ${e.message}`);
            } finally {
                setLoading(false);
            }
        };

        fetchCompare();
    }, [startCoord, endCoord, navigate]);

    // 지도 초기화 및 경로 그리기
    useEffect(() => {
        if (!mapRef.current || !compareData) return;

        if (mapInstanceRef.current) {
            mapInstanceRef.current.remove();
            mapInstanceRef.current = null;
        }

        const map = L.map(mapRef.current).setView([37.5665, 126.9780], 13);
        mapInstanceRef.current = map;

        // 베이스 레이어 (OSM)
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        // WMS 레이어 (범죄주의구간)
        const wmsLayer = L.tileLayer.wms("https://www.safemap.go.kr/openapi2/IF_0087_WMS", {
            layers: "A2SM_CRMNLHSPOT_TOT",
            styles: "A2SM_CrmnlHspot_Tot_Tot",
            format: "image/png",
            transparent: true,
            opacity: 0.4,
            serviceKey: "7F2ABSQ6-7F2A-7F2A-7F2A-7F2ABSQ61U"
        });
        wmsLayer.addTo(map);

        // --- 클러스터 생성 ---
        const createColoredCluster = (className) => {
            return L.markerClusterGroup({
                iconCreateFunction: function (cluster) {
                    const count = cluster.getChildCount();
                    return L.divIcon({
                        html: '<div>' + count + '</div>',
                        className: 'custom-cluster-icon ' + className,
                        iconSize: L.point(40, 40)
                    });
                }
            });
        };

        const cctvCluster = createColoredCluster('marker-cluster-cctv');
        const streetCluster = createColoredCluster('marker-cluster-street');
        const entCluster = createColoredCluster('marker-cluster-ent');
        const policeCluster = L.markerClusterGroup('marker-cluster-poli');
        const convCluster = createColoredCluster('marker-cluster-conv');

        // 포인트 데이터 추가
        if (points.length > 0) {
            points.forEach(p => {
                let marker;
                const popupContent = `<div style="text-align:center"><b>${p.name}</b><br/><span style="color:#666">${p.category}</span></div>`;

                if (p.category === 'cctv') marker = L.marker([p.lat, p.lng], { icon: cctvIcon }).bindPopup(popupContent);
                else if (p.category === 'streetlight') marker = L.marker([p.lat, p.lng], { icon: streetIcon }).bindPopup(popupContent);
                else if (p.category === 'police') marker = L.marker([p.lat, p.lng], { icon: policeIcon }).bindPopup(popupContent);
                else if (p.category === 'convenience') marker = L.marker([p.lat, p.lng], { icon: convIcon }).bindPopup(popupContent);
                else if (p.category === 'entertainment') marker = L.marker([p.lat, p.lng], { icon: entIcon }).bindPopup(popupContent);

                if (marker) {
                    if (p.category === 'cctv') cctvCluster.addLayer(marker);
                    else if (p.category === 'streetlight') streetCluster.addLayer(marker);
                    else if (p.category === 'police') policeCluster.addLayer(marker);
                    else if (p.category === 'convenience') convCluster.addLayer(marker);
                    else if (p.category === 'entertainment') entCluster.addLayer(marker);
                }
            });
        }

        // 클러스터 추가
        map.addLayer(cctvCluster);
        map.addLayer(streetCluster);
        map.addLayer(policeCluster);
        map.addLayer(convCluster);
        map.addLayer(entCluster);

        // 레이어 컨트롤
        const overlays = {
            "CCTV (파랑)": cctvCluster,
            "가로등 (노랑)": streetCluster,
            "경찰서 (보라)": policeCluster,
            "편의점 (초록)": convCluster,
            "유흥업소 (빨강)": entCluster,
            "범죄주의구간(WMS)": wmsLayer
        };
        L.control.layers(null, overlays, { collapsed: false }).addTo(map);

        // --- 경로 그리기 ---
        const allCoords = [];

        // 최단 경로 (파란색, 점선)
        if (compareData.shortest && (activeRoute === 'both' || activeRoute === 'shortest')) {
            const shortestCoords = compareData.shortest.path_coords.map(c => [c[0], c[1]]);
            allCoords.push(...shortestCoords);

            L.polyline(shortestCoords, {
                color: '#3b82f6',
                weight: 5,
                opacity: 0.9,
                dashArray: '10, 10'
            }).addTo(map).bindPopup(`<b>🔵 최단 경로</b><br/>거리: ${compareData.shortest.length.toFixed(0)}m<br/>안전점수: ${compareData.shortest.safety_score}점`);
        }

        // 안전 경로 (초록색, 실선)
        if (compareData.safest && (activeRoute === 'both' || activeRoute === 'safest')) {
            const safestCoords = compareData.safest.path_coords.map(c => [c[0], c[1]]);
            allCoords.push(...safestCoords);

            L.polyline(safestCoords, {
                color: '#22c55e',
                weight: 6,
                opacity: 0.9
            }).addTo(map).bindPopup(`<b>🟢 안전 경로</b><br/>거리: ${compareData.safest.length.toFixed(0)}m<br/>안전점수: ${compareData.safest.safety_score}점`);
        }

        // 출발/도착 마커
        if (allCoords.length > 0) {
            const startPoint = allCoords[0];
            const endPoint = compareData.safest?.path_coords?.slice(-1)[0] || compareData.shortest?.path_coords?.slice(-1)[0];

            L.marker(startPoint, {
                icon: L.divIcon({
                    className: '',
                    html: '<div style="background:#22c55e;color:white;padding:5px 10px;border-radius:20px;font-weight:bold;font-size:12px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.3);transform:translate(-50%,-100%);">출발</div>',
                    iconSize: null, iconAnchor: [0, 0]
                })
            }).addTo(map);

            if (endPoint) {
                L.marker([endPoint[0], endPoint[1]], {
                    icon: L.divIcon({
                        className: '',
                        html: '<div style="background:#ef4444;color:white;padding:5px 10px;border-radius:20px;font-weight:bold;font-size:12px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.3);transform:translate(-50%,-100%);">도착</div>',
                        iconSize: null, iconAnchor: [0, 0]
                    })
                }).addTo(map);
            }

            map.fitBounds(allCoords, { padding: [50, 50] });
        }

        return () => {
            if (mapInstanceRef.current) {
                mapInstanceRef.current.remove();
                mapInstanceRef.current = null;
            }
        };
    }, [compareData, activeRoute, points]);

    const getTimeInfo = () => {
        if (!compareData) return '';
        const hour = compareData.current_hour;
        const streetlightOn = compareData.streetlight_on;
        return `${hour}시 (가로등: ${streetlightOn ? 'ON 🌙' : 'OFF ☀️'})`;
    };

    return (
        <div className="container">
            <style>{clusterStyles}</style>
            <div className="card result-card">
                <div className="header-simple">
                    <button className="back-button" onClick={() => navigate(-1)}>←</button>
                    <h2 className="title">🛡️ 경로 비교</h2>
                </div>

                <div className="map-area-result">
                    {loading ? (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '10px' }}>
                            <div className="spinner" style={{ width: 30, height: 30, border: '4px solid #f3f3f3', borderTop: '4px solid #3498db', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                            <style>{`@keyframes spin {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}`}</style>
                            <div>경로를 계산하고 있습니다...</div>
                        </div>
                    ) : errorMsg ? (
                        <div style={{ padding: 20, color: 'red', textAlign: 'center' }}>
                            {errorMsg}
                            <br />
                            <button className="btn btn-outline small" style={{ marginTop: 10 }} onClick={() => navigate(-1)}>뒤로 가기</button>
                        </div>
                    ) : (
                        <div ref={mapRef} style={{ width: '100%', height: '100%' }} />
                    )}
                </div>

                {/* 경로 전환 버튼 */}
                {compareData && (
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', margin: '10px 0' }}>
                        <button
                            className={`btn small ${activeRoute === 'both' ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setActiveRoute('both')}
                        >
                            둘 다 보기
                        </button>
                        <button
                            className={`btn small ${activeRoute === 'shortest' ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setActiveRoute('shortest')}
                            style={{ borderColor: '#3b82f6', color: activeRoute === 'shortest' ? 'white' : '#3b82f6', background: activeRoute === 'shortest' ? '#3b82f6' : 'transparent' }}
                        >
                            🔵 최단
                        </button>
                        <button
                            className={`btn small ${activeRoute === 'safest' ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setActiveRoute('safest')}
                            style={{ borderColor: '#22c55e', color: activeRoute === 'safest' ? 'white' : '#22c55e', background: activeRoute === 'safest' ? '#22c55e' : 'transparent' }}
                        >
                            🟢 안전
                        </button>
                    </div>
                )}

                {/* 결과 카드 */}
                {compareData && (
                    <div className="info-card" style={{ padding: '15px' }}>
                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                            ⏰ 현재 시간: {getTimeInfo()}
                        </div>

                        <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
                            <div style={{ flex: 1, background: '#eff6ff', borderRadius: '10px', padding: '12px', border: '2px solid #3b82f6' }}>
                                <div style={{ fontWeight: 'bold', color: '#3b82f6', marginBottom: '8px' }}>🔵 최단 경로</div>
                                <div style={{ fontSize: '18px', fontWeight: 'bold' }}>{compareData.shortest.length.toFixed(0)}m</div>
                                <div style={{ fontSize: '14px', color: '#666' }}>안전점수: {compareData.shortest.safety_score}점</div>
                            </div>

                            <div style={{ flex: 1, background: '#f0fdf4', borderRadius: '10px', padding: '12px', border: '2px solid #22c55e' }}>
                                <div style={{ fontWeight: 'bold', color: '#22c55e', marginBottom: '8px' }}>🟢 안전 경로</div>
                                <div style={{ fontSize: '18px', fontWeight: 'bold' }}>{compareData.safest.length.toFixed(0)}m</div>
                                <div style={{ fontSize: '14px', color: '#666' }}>안전점수: {compareData.safest.safety_score}점</div>
                            </div>
                        </div>

                        <div style={{ background: '#fef3c7', borderRadius: '10px', padding: '12px', textAlign: 'center' }}>
                            <span style={{ color: '#d97706', fontWeight: 'bold' }}>
                                +{compareData.length_difference.toFixed(0)}m 더 걸어서
                            </span>
                            <span style={{ color: '#059669', fontWeight: 'bold', marginLeft: '8px' }}>
                                +{compareData.safety_improvement}점 더 안전!
                            </span>
                        </div>
                    </div>
                )}

                <div className="bottom-actions">
                    <button className="btn btn-outline small" onClick={() => navigate('/main')}>다시 검색</button>
                    <button className="btn btn-primary small">길안내 시작</button>
                </div>
            </div>

            {/* Chatbot Widget */}
            <ChatWidget
                currentLat={startCoord?.lat || 37.5665}
                currentLng={startCoord?.lon || 126.9780}
                onMoveTo={(lat, lng) => {
                    if (mapInstanceRef.current) {
                        mapInstanceRef.current.setView([lat, lng], 16);
                    }
                }}
                onDrawRoute={(routeData) => {
                    if (mapInstanceRef.current && routeData?.coordinates) {
                        const coords = routeData.coordinates.map(c => [c[1], c[0]]);
                        L.polyline(coords, {
                            color: '#ff6b6b',
                            weight: 5,
                            opacity: 0.8
                        }).addTo(mapInstanceRef.current);
                        mapInstanceRef.current.fitBounds(coords, { padding: [50, 50] });
                    }
                }}
            />
        </div>
    );
};

export default ResultPage;
