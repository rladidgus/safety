// src/hooks/useRouteCompare.js
import { useState, useCallback } from 'react';

const API_URL = 'http://localhost:8000';

/**
 * 경로 비교 API 호출 훅
 */
export default function useRouteCompare() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    const compareRoutes = useCallback(async (startLat, startLon, endLat, endLon) => {
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch(`${API_URL}/api/route/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    start_lat: startLat,
                    start_lon: startLon,
                    end_lat: endLat,
                    end_lon: endLon
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || '경로를 찾을 수 없습니다.');
            }

            const data = await response.json();
            setResult(data);
            return data;
        } catch (e) {
            setError(e.message);
            throw e;
        } finally {
            setLoading(false);
        }
    }, []);

    const reset = useCallback(() => {
        setLoading(false);
        setError(null);
        setResult(null);
    }, []);

    return { loading, error, result, compareRoutes, reset };
}
