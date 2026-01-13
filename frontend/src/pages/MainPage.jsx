// src/pages/MainPage.jsx
import React, { useState, useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import 'leaflet.markercluster';
import ChatWidget from '../components/ChatWidget';
import '../index.css';

const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search';

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

const MainPage = () => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);

  const [startPoint, setStartPoint] = useState('');
  const [endPoint, setEndPoint] = useState('');
  const [startCoord, setStartCoord] = useState(null);
  const [endCoord, setEndCoord] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');


  // 결과 관련 상태
  const [compareData, setCompareData] = useState(null);
  const [points, setPoints] = useState([]);
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
    setCompareData(null);

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

      // API 호출
      const res = await fetch('http://localhost:8000/api/route/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_lat: start.lat,
          start_lon: start.lon,
          end_lat: end.lat,
          end_lon: end.lon
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        setError(errData.detail || '경로를 찾을 수 없습니다.');
        return;
      }

      const data = await res.json();
      console.log('Compare API response:', data);
      setCompareData(data);

    } catch (e) {
      console.error('Error:', e);
      setError(`오류가 발생했습니다: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

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

    // 클러스터 추가 (기본적으로는 끔 - 사용자가 선택하도록)
    // map.addLayer(cctvCluster);
    // map.addLayer(streetCluster);
    // map.addLayer(policeCluster);
    // map.addLayer(convCluster);
    // map.addLayer(entCluster);

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
      }).addTo(map).bindPopup(`<b>🔵 최단 경로</b>`);
    }

    // 안전 경로 (초록색, 실선)
    if (compareData.safest && (activeRoute === 'both' || activeRoute === 'safest')) {
      const safestCoords = compareData.safest.path_coords.map(c => [c[0], c[1]]);
      allCoords.push(...safestCoords);

      L.polyline(safestCoords, {
        color: '#22c55e',
        weight: 6,
        opacity: 0.9
      }).addTo(map).bindPopup(`<b>🟢 안전 경로</b>`);
    }

    // 출발/도착 마커
    if (allCoords.length > 0) {
      const startMarkerPoint = allCoords[0];
      const endMarkerPoint = compareData.safest?.path_coords?.slice(-1)[0] || compareData.shortest?.path_coords?.slice(-1)[0];

      L.marker(startMarkerPoint, {
        icon: L.divIcon({
          className: '',
          html: '<div style="background:#22c55e;color:white;padding:5px 10px;border-radius:20px;font-weight:bold;font-size:12px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.3);transform:translate(-50%,-100%);">출발</div>',
          iconSize: null, iconAnchor: [0, 0]
        })
      }).addTo(map);

      if (endMarkerPoint) {
        L.marker([endMarkerPoint[0], endMarkerPoint[1]], {
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

  return (
    <div className="container">
      <style>{clusterStyles}</style>
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
                setStartCoord(null);
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
                setEndCoord(null);
              }}
            />
          </div>


        </div>

        {error && (
          <div style={{ color: 'red', textAlign: 'center', margin: '10px 20px', fontSize: '14px' }}>
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

        {/* 지도 영역 */}
        <div className="map-placeholder-main" style={{ marginTop: '20px', position: 'relative' }}>
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '10px' }}>
              <div className="spinner" style={{ width: 30, height: 30, border: '4px solid #f3f3f3', borderTop: '4px solid #3498db', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
              <style>{`@keyframes spin {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}`}</style>
              <div>경로를 계산하고 있습니다...</div>
            </div>
          ) : compareData ? (
            <>
              <div ref={mapRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />
              {/* Chatbot Widget - 지도 오른쪽 하단 */}

            </>
          ) : (
            <div className="map-area-text">
              <span>📍 출발지와 목적지를 입력하고 검색하세요</span>
            </div>
          )}
        </div>

        {/* 경로 전환 버튼 - 지도 아래 */}
        {compareData && (
          <div style={{ display: 'flex', gap: '6px', justifyContent: 'center', margin: '10px 20px' }}>
            <button
              className={`btn xs ${activeRoute === 'both' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setActiveRoute('both')}
            >
              둘 다 보기
            </button>
            <button
              className={`btn xs ${activeRoute === 'shortest' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setActiveRoute('shortest')}
              style={{ borderColor: '#3b82f6', color: activeRoute === 'shortest' ? 'white' : '#3b82f6', background: activeRoute === 'shortest' ? '#3b82f6' : 'transparent' }}
            >
              🔵 최단
            </button>
            <button
              className={`btn xs ${activeRoute === 'safest' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setActiveRoute('safest')}
              style={{ borderColor: '#22c55e', color: activeRoute === 'safest' ? 'white' : '#22c55e', background: activeRoute === 'safest' ? '#22c55e' : 'transparent' }}
            >
              🟢 안전
            </button>
          </div>
        )}

        {/* AI Analysis Display */}
        {compareData && (
          <div style={{ padding: '15px', background: '#f8f9fa', borderRadius: '12px', margin: '0 20px 20px 20px', border: '1px solid #e9ecef' }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', gap: '6px' }}>
              <span style={{ fontSize: '18px' }}>🤖</span>
              <span style={{ fontWeight: 'bold', color: '#1f2937', fontSize: '14px' }}>AI 안전 분석 결과</span>
            </div>
            <p style={{ fontSize: '13px', color: '#4b5563', lineHeight: '1.5', margin: 0 }}>
              {activeRoute === 'safest'
                ? compareData.safest.ai_analysis || "분석 정보가 없습니다."
                : activeRoute === 'shortest'
                  ? compareData.shortest.ai_analysis || "분석 정보가 없습니다."
                  : compareData.safest.ai_analysis // 'both'일 때는 안전 경로 분석 우선
              }
            </p>
          </div>
        )}

        {/* 하단 버튼 */}
        {compareData && (
          <div className="bottom-actions" style={{ marginTop: '15px', display: 'flex', gap: '10px', justifyContent: 'center' }}>
            <button
              className="btn btn-outline small"
              onClick={() => {
                setCompareData(null);
                setStartPoint('');
                setEndPoint('');
                setStartCoord(null);
                setEndCoord(null);
                setError('');
              }}
            >
              다시 검색
            </button>
            <button className="btn btn-primary small">길안내 시작</button>
          </div>
        )}

        {/* Chatbot Widget - Always Visible */}
        <ChatWidget
          currentLat={startCoord?.lat || 37.5665}
          currentLng={startCoord?.lon || 126.9780}
          onMoveTo={(lat, lng) => {
            if (mapInstanceRef.current) {
              mapInstanceRef.current.setView([lat, lng], 16);
            }
          }}
          onDrawRoute={(routeData) => {
            if (routeData?.coordinates) {
              // GeoJSON [lng, lat] -> Leaflet [lat, lng]
              const coords = routeData.coordinates.map(c => [c[1], c[0]]);
              if (mapInstanceRef.current) {
                L.polyline(coords, {
                  color: '#ff6b6b',
                  weight: 5,
                  opacity: 0.8
                }).addTo(mapInstanceRef.current);
                mapInstanceRef.current.fitBounds(coords, { padding: [50, 50] });
              }
            }
          }}
        />

      </div>

    </div>
  );
};

export default MainPage;
