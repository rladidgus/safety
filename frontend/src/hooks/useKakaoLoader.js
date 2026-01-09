import { useState, useEffect } from "react";

const KAKAO_KEY = "cbfe84d814e1b43c96395c085b0dc363";

export default function useKakaoLoader() {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        // 이미 로드되어 있다면
        if (window.kakao && window.kakao.maps) {
            setLoading(false);
            return;
        }

        // 이미 스크립트가 문서에 있다면 (로드 중이라면)
        const existingScript = document.getElementById("kakao-sdk");
        if (existingScript) {
            existingScript.addEventListener("load", () => setLoading(false));
            existingScript.addEventListener("error", (e) => setError(e));
            return;
        }

        // 스크립트 새로 생성
        const script = document.createElement("script");
        script.id = "kakao-sdk";
        script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${KAKAO_KEY}&libraries=services,clusterer,drawing&autoload=false`;
        script.async = true;

        script.onload = () => {
            // 로드 완료 후 kakao.maps.load 호출하여 확실히 초기화
            window.kakao.maps.load(() => {
                setLoading(false);
            });
        };

        script.onerror = (e) => {
            setError(e);
            setLoading(false);
        };

        document.head.appendChild(script);
    }, []);

    return { loading, error };
}
