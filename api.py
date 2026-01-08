"""
안전 경로 API 서버 (FastAPI)
- API 엔드포인트 제공
- 프론트엔드 정적 파일 서빙
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Tuple
import sys
from pathlib import Path

# 프로젝트 경로 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from route_finder import (
    load_graph, find_shortest_path, find_safest_path, 
    compare_routes, find_nearest_node, get_path_coords
)

app = FastAPI(
    title="안심 길 안내 API",
    description="서울시 안전 경로 탐색 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 그래프 로드 (서버 시작 시 1회)
G = None

@app.on_event("startup")
async def startup_event():
    global G
    try:
        G = load_graph()
        print(f"✅ 그래프 로드: {G.number_of_nodes():,} 노드, {G.number_of_edges():,} 엣지")
    except FileNotFoundError:
        print("⚠️ 그래프 파일 없음. python src/graph_builder.py 실행 필요")


class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


class RouteResult(BaseModel):
    path_coords: List[List[float]]
    length: float
    safety_score: int
    node_count: int
    route_type: str


class CompareResult(BaseModel):
    shortest: RouteResult
    safest: RouteResult
    length_difference: float
    length_difference_percent: float
    safety_improvement: int


# ============================================
# API 엔드포인트
# ============================================

@app.get("/api/health")
async def health_check():
    if G is None:
        return {"status": "error", "message": "그래프 로드 안됨"}
    return {
        "status": "ok",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "hour": G.graph.get('hour', 'N/A'),
        "streetlight_on": G.graph.get('streetlight_on', False)
    }


@app.post("/api/route", response_model=CompareResult)
async def search_route(request: RouteRequest):
    if G is None:
        raise HTTPException(status_code=500, detail="그래프가 로드되지 않았습니다.")
    
    origin = (request.start_lat, request.start_lon)
    destination = (request.end_lat, request.end_lon)
    
    result = compare_routes(G, origin, destination)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    shortest_coords = get_path_coords(G, result['shortest']['path'])
    safest_coords = get_path_coords(G, result['safest']['path'])
    
    return CompareResult(
        shortest=RouteResult(
            path_coords=[[c[0], c[1]] for c in shortest_coords],
            length=result['shortest']['length'],
            safety_score=result['shortest']['avg_safety_score'],
            node_count=len(result['shortest']['path']),
            route_type="shortest"
        ),
        safest=RouteResult(
            path_coords=[[c[0], c[1]] for c in safest_coords],
            length=result['safest']['length'],
            safety_score=result['safest']['avg_safety_score'],
            node_count=len(result['safest']['path']),
            route_type="safest"
        ),
        length_difference=result['length_difference'],
        length_difference_percent=result['length_difference_percent'],
        safety_improvement=result['safety_improvement']
    )


@app.get("/api/graph-info")
async def graph_info():
    if G is None:
        raise HTTPException(status_code=500, detail="그래프가 로드되지 않았습니다.")
    
    nodes = list(G.nodes())[:1000]
    lats = [G.nodes[n].get('lat', 0) for n in nodes]
    lons = [G.nodes[n].get('lon', 0) for n in nodes]
    
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "bounds": {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        },
        "center": {
            "lat": sum(lats) / len(lats),
            "lon": sum(lons) / len(lons)
        }
    }


# ============================================
# 프론트엔드 서빙
# ============================================

@app.get("/")
async def serve_frontend():
    """메인 페이지 서빙"""
    frontend_path = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "안심 길 안내 API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🚀 안심 길 안내 서버 시작")
    print("=" * 50)
    print("📍 웹 UI: http://localhost:8000")
    print("📍 API 문서: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
