import os
import logging
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt
import networkx as nx
import osmnx as ox
import math
from pathlib import Path
import pandas as pd # Added pandas
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Safety Module Imports ---
# Assuming these are now in the same package (backend/app/)
try:
    from .graph_builder import load_graph, main as build_graph_main
    from .route_finder import find_safest_path, find_shortest_path
except ImportError:
    # Fallback if run as script
    from graph_builder import load_graph, main as build_graph_main
    from route_finder import find_safest_path, find_shortest_path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup (from miniPJ01.py) ---
# Saving DB in the parent directory (backend/) to keep it persistent
# We assume we are running likely from 'backend/' folder
SQLALCHEMY_DATABASE_URL = "sqlite:///./user.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# --- Pydantic Models ---
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class RouteRequest(BaseModel):
    start_address: str
    end_address: str
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    mode: str = "safe" # 'safe' or 'shortest'

class LatLng(BaseModel):
    lat: float
    lng: float

class RouteResponse(BaseModel):
    path: List[LatLng]
    distance: float
    duration: float # estimated
    safety_score: float

# --- Security Helper Functions ---
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Global Graph Variable ---
G: Optional[nx.MultiDiGraph] = None

# --- FastAPI App ---
app = FastAPI(title="Safe Route Navigation Service")

# CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Import safety modules
from .graph_builder import load_graph, build_weighted_graph_ml
from .route_finder import find_safest_path, find_shortest_path

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global G
    try:
        logger.info("Loading safety graph...")
        # Check if graph exists, if not, try to build it (or warn)
        try:
            G = load_graph("safety_graph")
            logger.info(f"Graph loaded successfully. Nodes: {G.number_of_nodes()}")
        except Exception:
            logger.warning("Graph file not found. Building default graph (Gangnam-gu)... This might take time.")
            G = build_graph_main("Gangnam-gu, Seoul, South Korea") 
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to Safety Route API"}

# --- Auth Endpoints ---

@app.post("/api/signup", status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자명입니다.")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "회원가입 성공", "username": new_user.username}

@app.post("/api/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="아이디 또는 비밀번호가 잘못되었습니다.")
    
    return {"message": "로그인 성공", "username": db_user.username}

# --- Route Endpoint ---

@app.post("/api/route/safe", response_model=RouteResponse)
def get_route(req: RouteRequest):
    global G
    if G is None:
        raise HTTPException(status_code=503, detail="Graph not initialized yet. Please wait.")
    
    try:
        # 1. Geocoding or Use Provided Coordinates
        logger.info(f"Request: {req.start_address} -> {req.end_address}")
        
        if req.start_lat and req.start_lon and req.end_lat and req.end_lon:
            start_lat, start_lon = req.start_lat, req.start_lon
            end_lat, end_lon = req.end_lat, req.end_lon
            logger.info(f"Using provided coordinates: {start_lat},{start_lon} -> {end_lat},{end_lon}")
        else:
            # Fallback to backend geocoding (OSMnx/Nominatim)
            logger.info("Coordinates not provided, attempting backend geocoding...")
            try:
                start_lat, start_lon = ox.geocode(req.start_address)
                end_lat, end_lon = ox.geocode(req.end_address)
            except Exception as e:
                logger.error(f"Geocoding failed: {e}")
                raise HTTPException(status_code=400, detail=f"주소를 찾을 수 없습니다: {e}")

        origin = (start_lat, start_lon)
        destination = (end_lat, end_lon)

        # 2. Find Path
        if req.mode == "shortest":
            result = find_shortest_path(G, origin, destination)
        else:
            result = find_safest_path(G, origin, destination)
        
        if 'error' in result:
             raise HTTPException(status_code=404, detail=result['error'])

        # 3. Format Response
        path_nodes = result['path']
        path_latlngs = []
        for node in path_nodes:
            # G.nodes[node]['y'] is lat, ['x'] is lon
            lat = G.nodes[node]['y']
            lng = G.nodes[node]['x']
            path_latlngs.append(LatLng(lat=lat, lng=lng))
        
        # Estimate duration (walking speed ~5km/h = ~1.4 m/s)
        distance_meters = result['length']
        duration_seconds = distance_meters / 1.4

        return RouteResponse(
            path=path_latlngs,
            distance=distance_meters,
            duration=duration_seconds,
            safety_score=result.get('avg_safety_score', 50.0)
        )

    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Points API for Dgdisplay Features ---
# backend/app/main.py -> backend/app -> backend -> src -> data -> processed
# Adjust based on step 269: PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# If main.py is in backend/app/, parent.parent is backend. src is in backend.
BASE_DIR = Path(__file__).resolve().parent.parent # backend/
DATA_DIR = BASE_DIR / "src" / "data" / "processed"

def load_facility_features(filename: str, category: str):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        # Try finding it relative to project root if not found
        # fallback for different structure
        return []
    
    try:
        df = pd.read_csv(filepath)
        features = []
        for _, row in df.iterrows():
            lon = row.get("lon") or row.get("longitude") or row.get("x")
            lat = row.get("lat") or row.get("latitude") or row.get("y")
            name = row.get("name") or row.get("시설명") or ""
            
            if pd.isna(lat) or pd.isna(lon):
                continue
                
            features.append({
                "lat": float(lat),
                "lng": float(lon),
                "category": category,
                "name": str(name)
            })
        return features
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return []

@app.get("/api/points")
def get_all_points():
    logger.info("Fetching all safety points...")
    all_points = []
    
    # Check if DATA_DIR exists
    if not DATA_DIR.exists():
        logger.warning(f"Data directory not found at {DATA_DIR}")
        return {"points": []}

    all_points.extend(load_facility_features("cctv.csv", "cctv"))
    all_points.extend(load_facility_features("streetlights.csv", "streetlight"))
    all_points.extend(load_facility_features("convenience_stores.csv", "convenience"))
    all_points.extend(load_facility_features("entertainment_danger.csv", "entertainment"))
    
    logger.info(f"Returning {len(all_points)} features for map")
    return {"points": all_points}

if __name__ == "__main__":
    import uvicorn
    # If run directly, assume "app" package context might be tricky, 
    # but we rely on relative imports handling or running from parent.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)