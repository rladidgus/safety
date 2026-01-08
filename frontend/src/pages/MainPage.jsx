// import React, { useEffect, useState } from 'react';
// import { useLocation, useNavigate } from 'react-router-dom';
// import MapContainer from '../components/MapContainer';
// import '../index.css';

// const MainPage = () => {
//     const navigate = useNavigate();
//     const [startPoint, setStartPoint] = useState('');
//     const [endPoint, setEndPoint] = useState('');

    const handleSafeSearch = () => {
        navigate('/result', { state: { start: startPoint, end: endPoint, type: 'safe' } });
    };

//     const handleGeneralSearch = () => {
//         navigate('/result', { state: { start: startPoint, end: endPoint, type: 'general' } });
//     };

//     return (
//         <div className="container">
//             <div className="card">
//                 <div className="header">
//                     <h1 className="service-title">서비스명</h1>
//                     <button className="menu-button">☰</button>
//                 </div>

//                 <div className="input-group">
//                     <div className="input-wrapper">
//                         <span className="icon">📍</span>
//                         <input
//                             type="text"
//                             placeholder="출발지를 입력하세요"
//                             value={startPoint}
//                             onChange={(e) => setStartPoint(e.target.value)}
//                         />
//                         <button className="check-button">✓</button>
//                     </div>
//                     <div className="input-wrapper">
//                         <span className="icon">📍</span>
//                         <input
//                             type="text"
//                             placeholder="목적지를 입력하세요"
//                             value={endPoint}
//                             onChange={(e) => setEndPoint(e.target.value)}
//                         />
//                         <button className="clear-button">✖</button>
//                     </div>
//                 </div>

//                 <div className="button-group">
//                     <button className="btn btn-outline" onClick={handleGeneralSearch}>일반 경로 탐색</button>
//                     <button className="btn btn-primary" onClick={handleSafeSearch}>안전 경로 탐색</button>
//                 </div>

//                 <div className="map-placeholder-main">
//                     <div className="map-area-text">
//                         <span>📍 Map Area</span>
//                     </div>
//                 </div>

//                 <div className="bottom-link">
//                     <a href="#">안전 경로란?</a>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default MainPage;

// src/pages/MainPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../index.css';

const MainPage = () => {
  const navigate = useNavigate();
  const [startPoint, setStartPoint] = useState('');
  const [endPoint, setEndPoint] = useState('');

  const handleSafeSearch = () => {
    navigate('/result', {
      state: { start: startPoint, end: endPoint, type: 'safe' },
    });
  };

  const handleGeneralSearch = () => {
    navigate('/result', {
      state: { start: startPoint, end: endPoint, type: 'general' },
    });
  };

  return (
    <div className="container">
      <div className="card">
        <div className="header">
          <h1 className="service-title">서비스명</h1>
          <button className="menu-button">☰</button>
        </div>

        <div className="input-group">
          <div className="input-wrapper">
            <span className="icon">📍</span>
            <input
              type="text"
              placeholder="출발지를 입력하세요"
              value={startPoint}
              onChange={(e) => setStartPoint(e.target.value)}
            />
            <button className="check-button">✓</button>
          </div>
          <div className="input-wrapper">
            <span className="icon">📍</span>
            <input
              type="text"
              placeholder="목적지를 입력하세요"
              value={endPoint}
              onChange={(e) => setEndPoint(e.target.value)}
            />
            <button className="clear-button">✖</button>
          </div>
        </div>

        <div className="button-group">
          <button className="btn btn-outline" onClick={handleGeneralSearch}>
            일반 경로 탐색
          </button>
          <button className="btn btn-primary" onClick={handleSafeSearch}>
            안전 경로 탐색
          </button>
        </div>

        <div className="map-placeholder-main">
          <div className="map-area-text">
            <span>📍 Map Area</span>
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
