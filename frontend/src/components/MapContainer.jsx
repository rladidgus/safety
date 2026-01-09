// src/components/MapContainer.jsx
import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import 'leaflet.markercluster';

// Fix Leaflet's default icon issue
import iconManager from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// --- Custom Icons ---
// Safety Facilities
const cctvIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
const streetIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-gold.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
const entIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
const policeIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-violet.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
const convIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

// Start/End Icons
const startMarkerIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-grey.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
const endMarkerIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-black.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

// --- Custom Cluster CSS (injected via style tag) ---
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
    .marker-cluster-cctv {
        background-color: rgba(59, 130, 246, 0.8); /* Blue */
    }
    .marker-cluster-cctv div {
        background-color: transparent;
    }
    .marker-cluster-street {
        background-color: rgba(234, 179, 8, 0.85); /* Yellow/Gold */
        color: #fff;
    }
    .marker-cluster-ent {
        background-color: rgba(239, 68, 68, 0.85); /* Red */
    }
    /* Default Overrides */
    .custom-cluster-icon div {
        width: 36px;
        height: 36px;
        margin: 0;
        text-align: center;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
`;

const MapContainer = ({ routeData }) => {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const [points, setPoints] = useState([]);

  // 1. Fetch Points
  useEffect(() => {
    fetch('http://localhost:8000/api/points')
      .then(res => res.json())
      .then(data => {
        if (data.points) setPoints(data.points);
      })
      .catch(err => console.error("Error fetching points:", err));
  }, []);

  // 2. Initialize Map & Layers
  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Cleanup existing map
    if (mapInstanceRef.current) {
      mapInstanceRef.current.remove();
      mapInstanceRef.current = null;
    }

    // Create Map
    const map = L.map(mapContainerRef.current).setView([37.5665, 126.9780], 13);
    mapInstanceRef.current = map;

    // Base Layer (OSM)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    // WMS Layer (Crime Attention)
    const wmsLayer = L.tileLayer.wms("https://www.safemap.go.kr/openapi2/IF_0087_WMS", {
      layers: "A2SM_CRMNLHSPOT_TOT",
      styles: "A2SM_CrmnlHspot_Tot_Tot",
      format: "image/png",
      transparent: true,
      opacity: 0.5,
      serviceKey: "7F2ABSQ6-7F2A-7F2A-7F2A-7F2ABSQ61U",
      version: '1.1.1'
    });
    wmsLayer.addTo(map);

    // Helper: Create Custom Colored Cluster
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

    // Standard Clusters for others
    const policeCluster = L.markerClusterGroup();
    const convCluster = L.markerClusterGroup();

    // Populate Clusters
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

    // Add clusters to map
    map.addLayer(cctvCluster);
    map.addLayer(streetCluster);
    map.addLayer(policeCluster);
    map.addLayer(convCluster);
    map.addLayer(entCluster);

    // Layer Control
    const overlays = {
      "CCTV (파랑)": cctvCluster,
      "가로등 (노랑)": streetCluster,
      "경찰서": policeCluster,
      "편의점": convCluster,
      "유흥업소 (빨강)": entCluster,
      "범죄주의구간(WMS)": wmsLayer
    };
    L.control.layers(null, overlays, { collapsed: false }).addTo(map);

    // Route Polyline & Start/End Markers
    if (routeData && routeData.path && routeData.path.length > 0) {
      const latlngs = routeData.path.map(p => [p.lat, p.lng]);

      // Route Line
      const polyline = L.polyline(latlngs, { color: 'blue', weight: 5, opacity: 0.7 }).addTo(map);
      map.fitBounds(polyline.getBounds(), { padding: [50, 50] });

      // Start Marker (First Point)
      const startPoint = latlngs[0];
      L.marker(startPoint, { icon: startMarkerIcon })
        .addTo(map)
        .bindPopup("<b>출발지</b>");
      // .openPopup(); // Optional: open by default? User didn't specify.

      // End Marker (Last Point)
      const endPoint = latlngs[latlngs.length - 1];
      L.marker(endPoint, { icon: endMarkerIcon })
        .addTo(map)
        .bindPopup("<b>도착지</b>");
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [points, routeData]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <style>{clusterStyles}</style>
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

export default MapContainer;
