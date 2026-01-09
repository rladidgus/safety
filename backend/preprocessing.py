"""
데이터 전처리 모듈
- CSV 파일 로드
- 좌표 변환 (TM → WGS84, 주소 → 좌표)
- 안전/위험 점수 피처 생성

데이터 파일:
- 서울시 가로등 위치 정보.csv (안전 요소)
- 서울시 안심이 CCTV 연계 현황.csv (안전 요소)
- 경찰청_서울 경찰관서 현황_20231231.csv (안전 요소 - 지오코딩 필요)
- 서울시 유흥주점영업 인허가 정보.csv (위험 요소 - TM좌표 변환 필요)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 좌표 변환용
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("⚠️ pyproj 미설치. TM좌표 변환을 위해 'pip install pyproj'를 실행해주세요.")

# 지오코딩용 (주소 → 좌표)
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("⚠️ geopy 미설치. 주소→좌표 변환을 위해 'pip install geopy'를 실행해주세요.")


# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_streetlights() -> pd.DataFrame:
    """
    가로등 데이터 로드 (안전 요소)
    컬럼: 가로등관리번호, 위도, 경도
    """
    filepath = DATA_DIR / "서울시 가로등 위치 정보.csv"
    
    if not filepath.exists():
        print(f"❌ 파일 없음: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding='cp949')
    
    # 컬럼명 정리
    df.columns = ['관리번호', '위도', '경도']
    df = df.rename(columns={'위도': 'latitude', '경도': 'longitude'})
    
    # 유효한 좌표만 필터링
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] > 33) & (df['latitude'] < 43)]
    df = df[(df['longitude'] > 124) & (df['longitude'] < 132)]
    
    print(f"✅ 가로등 데이터 로드: {len(df)} 건")
    return df[['latitude', 'longitude']]


def load_streetlight_schedule() -> dict:
    """
    가로등 점소등 시간 데이터 로드
    
    Returns:
        dict: {'on_hour': 점등 평균 시간, 'off_hour': 소등 평균 시간}
    """
    filepath = DATA_DIR / "서울시 가로등 점소등 시간 현황.csv"
    
    if not filepath.exists():
        print("⚠️ 가로등 점소등 시간 파일 없음, 기본값 사용 (18:00~06:00)")
        return {'on_hour': 18, 'off_hour': 6}
    
    try:
        df = pd.read_csv(filepath, encoding='cp949')
        
        # 시간 파싱 (HHMMSS 형식 -> 시간)
        def parse_time(t):
            if pd.isna(t):
                return None
            t = int(t)
            return t // 10000  # 시간만 추출
        
        on_times = df['서울시 실제 점등시간(시분초)'].apply(parse_time).dropna()
        off_times = df['서울시 실제 소등시간(시분초)'].apply(parse_time).dropna()
        
        avg_on = int(on_times.mean()) if len(on_times) > 0 else 18
        avg_off = int(off_times.mean()) if len(off_times) > 0 else 6
        
        print(f"✅ 가로등 점소등 시간 로드: 점등 {avg_on}시, 소등 {avg_off}시 (평균)")
        
        return {'on_hour': avg_on, 'off_hour': avg_off}
        
    except Exception as e:
        print(f"⚠️ 가로등 점소등 시간 파일 오류: {e}")
        return {'on_hour': 18, 'off_hour': 6}


def load_cctv() -> pd.DataFrame:
    """
    CCTV 데이터 로드 (안전 요소)
    컬럼: 자치구명, 설치 위치명, 위도, 경도, CCTV 대수, 데이터 기준일
    """
    filepath = DATA_DIR / "서울시 안심이 CCTV 연계 현황.csv"
    
    if not filepath.exists():
        print(f"❌ 파일 없음: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding='cp949')
    
    # 컬럼명 정리 (원본 컬럼 순서 기반)
    df.columns = ['자치구명', '설치위치명', '위도', '경도', 'CCTV대수', '기준일']
    df = df.rename(columns={'위도': 'latitude', '경도': 'longitude'})
    
    # 유효한 좌표만 필터링
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] > 33) & (df['latitude'] < 43)]
    df = df[(df['longitude'] > 124) & (df['longitude'] < 132)]
    
    print(f"✅ CCTV 데이터 로드: {len(df)} 건")
    return df[['latitude', 'longitude', 'CCTV대수']]


def convert_tm_to_wgs84(x: float, y: float) -> tuple:
    """
    TM 좌표계(EPSG:5174)를 WGS84(EPSG:4326)로 변환
    
    Args:
        x: TM X 좌표
        y: TM Y 좌표
    
    Returns:
        (latitude, longitude) 튜플
    """
    if not PYPROJ_AVAILABLE:
        return None, None
    
    try:
        # 서울시 TM 좌표계 (중부원점 - EPSG:5174 또는 2097)
        transformer = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        
        # 유효성 검사
        if 33 < lat < 43 and 124 < lon < 132:
            return lat, lon
        
        # 다른 TM 좌표계 시도 (EPSG:5181 - GRS80 중부원점)
        transformer = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        
        if 33 < lat < 43 and 124 < lon < 132:
            return lat, lon
            
    except Exception as e:
        pass
    
    return None, None


def load_entertainment_venues() -> pd.DataFrame:
    """
    유흥주점 데이터 로드 (위험 요소)
    TM 좌표를 WGS84로 변환
    """
    filepath = DATA_DIR / "서울시 유흥주점영업 인허가 정보.csv"
    
    if not filepath.exists():
        print(f"❌ 파일 없음: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding='cp949', low_memory=False)
    
    # X, Y 좌표 컬럼 찾기placements
    x_col = None
    y_col = None
    for col in df.columns:
        if 'X' in col and '좌표' in col:
            x_col = col
        if 'Y' in col and '좌표' in col:
            y_col = col
    
    if x_col is None or y_col is None:
        # 컬럼명이 다를 수 있으니 인덱스로 접근 (보통 끝에서 10번째 정도)
        print("⚠️ 좌표 컬럼을 찾을 수 없습니다. 컬럼명 확인 필요.")
        print(f"   컬럼 목록: {list(df.columns)}")
        return pd.DataFrame()
    
    print(f"   X좌표 컬럼: {x_col}, Y좌표 컬럼: {y_col}")
    
    # 좌표 변환
    if not PYPROJ_AVAILABLE:
        print("❌ pyproj가 설치되어 있지 않아 TM→WGS84 변환 불가")
        return pd.DataFrame()
    
    print("🔄 TM → WGS84 좌표 변환 중...")
    
    coords = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="좌표변환"):
        try:
            x = float(row[x_col])
            y = float(row[y_col])
            if pd.notna(x) and pd.notna(y) and x > 0 and y > 0:
                lat, lon = convert_tm_to_wgs84(x, y)
                if lat and lon:
                    coords.append({'latitude': lat, 'longitude': lon})
        except:
            continue
    
    result = pd.DataFrame(coords)
    print(f"✅ 유흥주점 데이터 로드: {len(result)} 건 (변환 성공)")
    return result


def geocode_address(address: str, geolocator, geocode_func) -> tuple:
    """주소를 좌표로 변환"""
    try:
        location = geocode_func(address)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None, None


def load_police_stations() -> pd.DataFrame:
    """
    경찰관서 데이터 로드 (안전 요소)
    기존에 처리된 캐시 파일이 있으면 사용, 없으면 건너뛰기
    (지오코딩은 시간이 오래 걸리고 SSL 오류 발생 가능)
    """
    # 이미 처리된 캐시 파일 확인
    cached_file = PROCESSED_DIR / "police_stations.csv"
    if cached_file.exists():
        try:
            result = pd.read_csv(cached_file)
            print(f"✅ 경찰관서 데이터 로드 (캐시): {len(result)} 건")
            return result
        except:
            pass
    
    # 캐시 없으면 건너뛰기 (지오코딩 시간 오래 걸림)
    print("⚠️ 경찰관서 데이터 건너뜀 (지오코딩 필요, 별도 실행 권장)")
    return pd.DataFrame()


def load_convenience_stores() -> pd.DataFrame:
    """
    편의점 데이터 로드 (안전 요소 - 24시간 운영, 야간 밝음)
    컬럼: id, place_name, x(경도), y(위도) 등
    """
    filepath = DATA_DIR / "seoul_convenience_only_seoul.csv"
    
    if not filepath.exists():
        print(f"❌ 파일 없음: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # x = 경도(longitude), y = 위도(latitude)
    df = df.rename(columns={'y': 'latitude', 'x': 'longitude'})
    
    # 유효한 좌표만 필터링
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] > 33) & (df['latitude'] < 43)]
    df = df[(df['longitude'] > 124) & (df['longitude'] < 132)]
    
    print(f"✅ 편의점 데이터 로드: {len(df)} 건")
    return df[['latitude', 'longitude']]


def load_schools() -> pd.DataFrame:
    """
    학교 데이터 로드 (안전 요소 - 초/중/고)
    학교 주변은 어린이 보호구역으로 더 안전함
    """
    all_schools = []
    
    school_files = [
        "서울시 초등학교 기본정보.csv",
        "서울시 중학교 기본정보.csv",
        "서울시 고등학교 기본정보.csv"
    ]
    
    for filename in school_files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
        
        # 여러 인코딩 시도
        df = None
        for encoding in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except:
                continue
        
        if df is None:
            continue
        
        # 좌표 컬럼 찾기 (다양한 컬럼명 대응)
        lat_cols = [c for c in df.columns if '위도' in c or 'lat' in c.lower()]
        lon_cols = [c for c in df.columns if '경도' in c or 'lon' in c.lower()]
        
        if lat_cols and lon_cols:
            school_df = df[[lat_cols[0], lon_cols[0]]].copy()
            school_df.columns = ['latitude', 'longitude']
            school_df = school_df.dropna()
            
            # 유효한 좌표만
            school_df = school_df[(school_df['latitude'] > 33) & (school_df['latitude'] < 43)]
            school_df = school_df[(school_df['longitude'] > 124) & (school_df['longitude'] < 132)]
            
            all_schools.append(school_df)
    
    if all_schools:
        result = pd.concat(all_schools, ignore_index=True)
        print(f"✅ 학교 데이터 로드: {len(result)} 건 (초/중/고 통합)")
        return result
    
    print("⚠️ 학교 데이터 없음 (좌표 컬럼 없을 수 있음)")
    return pd.DataFrame()


def load_child_protection_zones() -> pd.DataFrame:
    """
    어린이 보호구역 데이터 로드 (안전 요소)
    """
    # xlsx 파일 찾기
    xlsx_files = list(DATA_DIR.glob("어린이*보호구역*.xlsx"))
    csv_files = list(DATA_DIR.glob("어린이*보호구역*.csv"))
    
    filepath = xlsx_files[0] if xlsx_files else (csv_files[0] if csv_files else None)
    
    if filepath is None:
        print("⚠️ 어린이 보호구역 파일 없음")
        return pd.DataFrame()
    
    try:
        if str(filepath).endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            # 여러 인코딩 시도
            for encoding in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except:
                    continue
        
        # 좌표 컬럼 찾기 (y좌표=위도, x좌표=경도)
        lat_cols = [c for c in df.columns if '위도' in c or 'lat' in c.lower() or c == 'y좌표']
        lon_cols = [c for c in df.columns if '경도' in c or 'lon' in c.lower() or c == 'x좌표']
        
        if lat_cols and lon_cols:
            result = df[[lat_cols[0], lon_cols[0]]].copy()
            result.columns = ['latitude', 'longitude']
            result = result.dropna()
            
            # 유효한 좌표만
            result = result[(result['latitude'] > 33) & (result['latitude'] < 43)]
            result = result[(result['longitude'] > 124) & (result['longitude'] < 132)]
            
            print(f"✅ 어린이 보호구역 로드: {len(result)} 건")
            return result
        else:
            print(f"⚠️ 어린이 보호구역 좌표 컬럼 없음 (컬럼: {list(df.columns)[:5]}...)")
    except Exception as e:
        print(f"⚠️ 어린이 보호구역 파일 오류: {e}")
    
    return pd.DataFrame()


def load_pedestrian_network() -> dict:
    """
    도보 네트워크 데이터 로드
    - 노드: 도로 교차점 좌표
    - 링크: 도로 구간 정보 (횡단보도, 공원, 터널, 교량 등)
    
    Returns:
        dict: {'nodes': DataFrame, 'links': DataFrame, 
               'crosswalk': DataFrame, 'park': DataFrame, 'tunnel': DataFrame}
    """
    filepath = DATA_DIR / "서울시 자치구별 도보 네트워크 공간정보.csv"
    
    if not filepath.exists():
        print(f"⚠️ 도보 네트워크 파일 없음: {filepath}")
        return {}
    
    print("   파일 로드 중 (대용량, 잠시 대기)...")
    
    try:
        df = pd.read_csv(filepath, encoding='cp949')
        print(f"   원본 데이터: {len(df):,} 행")
        
        # 노드 데이터 (좌표 포함)
        nodes = df[df['노드링크 유형'] == 'NODE'].copy()
        
        # WKT에서 좌표 추출 (POINT(lon lat) 형식)
        def parse_point_wkt(wkt):
            if pd.isna(wkt):
                return None, None
            try:
                # POINT(126.xxx 37.xxx) 형식 파싱
                coords = wkt.replace('POINT(', '').replace(')', '').split()
                lon, lat = float(coords[0]), float(coords[1])
                if 124 < lon < 132 and 33 < lat < 43:
                    return lat, lon
            except:
                pass
            return None, None
        
        # 노드 좌표 추출
        coords = nodes['노드 WKT'].apply(parse_point_wkt)
        nodes['latitude'] = coords.apply(lambda x: x[0])
        nodes['longitude'] = coords.apply(lambda x: x[1])
        nodes = nodes.dropna(subset=['latitude', 'longitude'])
        
        node_result = nodes[['노드 ID', 'latitude', 'longitude', '시군구명', '읍면동명']].copy()
        node_result.columns = ['node_id', 'latitude', 'longitude', 'district', 'dong']
        
        # 링크 데이터 (도로 구간)
        links = df[df['노드링크 유형'] == 'LINK'].copy()
        
        # 필요한 컬럼만 추출
        link_cols = ['링크 ID', '시작노드 ID', '종료노드 ID', '링크 길이', 
                     '시군구명', '읍면동명', '고가도로', '교량', '터널', 
                     '육교', '횡단보도', '공원,녹지', '건물내']
        links = links[link_cols].copy()
        links.columns = ['link_id', 'start_node', 'end_node', 'length',
                         'district', 'dong', 'elevated', 'bridge', 'tunnel',
                         'overpass', 'crosswalk', 'park', 'indoor']
        
        # 수치형 변환
        for col in ['elevated', 'bridge', 'tunnel', 'overpass', 'crosswalk', 'park', 'indoor']:
            links[col] = pd.to_numeric(links[col], errors='coerce').fillna(0).astype(int)
        
        # 특수 구간 추출 (위치 정보와 결합)
        crosswalk_links = links[links['crosswalk'] == 1].copy()
        park_links = links[links['park'] == 1].copy()
        tunnel_links = links[links['tunnel'] == 1].copy()
        
        print(f"   ✅ 노드: {len(node_result):,} 건")
        print(f"   ✅ 링크: {len(links):,} 건")
        print(f"   ✅ 횡단보도 구간: {len(crosswalk_links):,} 건")
        print(f"   ✅ 공원/녹지 구간: {len(park_links):,} 건")
        print(f"   ✅ 터널 구간: {len(tunnel_links):,} 건")
        
        return {
            'nodes': node_result,
            'links': links,
            'crosswalk_links': crosswalk_links,
            'park_links': park_links,
            'tunnel_links': tunnel_links
        }
        
    except Exception as e:
        print(f"⚠️ 도보 네트워크 파일 오류: {e}")
        return {}


def preprocess_all_data():
    """전체 데이터 전처리 파이프라인"""
    print("=" * 60)
    print("🚀 안심 길 안내 서비스 - 데이터 전처리 시작")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    # 1. 안전 요소 데이터 로드
    print("\n📍 [1/7] 가로등 데이터")
    streetlights = load_streetlights()
    
    print("\n📍 [2/7] CCTV 데이터")
    cctv = load_cctv()
    
    print("\n📍 [3/7] 경찰관서 데이터")
    police = load_police_stations()
    
    print("\n📍 [4/7] 편의점 데이터 (안전 요소)")
    convenience = load_convenience_stores()
    
    print("\n📍 [5/7] 학교 데이터 (안전 요소)")
    schools = load_schools()
    
    print("\n📍 [6/7] 어린이 보호구역 데이터 (안전 요소)")
    child_zones = load_child_protection_zones()
    
    # 2. 위험 요소 데이터 로드
    print("\n📍 [7/8] 유흥주점 데이터 (위험 요소)")
    entertainment = load_entertainment_venues()
    
    # 3. 도보 네트워크 데이터 로드
    print("\n📍 [8/8] 도보 네트워크 데이터")
    pedestrian = load_pedestrian_network()
    
    # 4. 전처리된 데이터 저장
    print("\n" + "=" * 60)
    print("💾 전처리 데이터 저장 중...")
    
    if len(streetlights) > 0:
        streetlights.to_csv(PROCESSED_DIR / "streetlights.csv", index=False)
        print(f"   ✅ streetlights.csv 저장 ({len(streetlights)} 건)")
    
    if len(cctv) > 0:
        cctv.to_csv(PROCESSED_DIR / "cctv.csv", index=False)
        print(f"   ✅ cctv.csv 저장 ({len(cctv)} 건)")
    
    if len(police) > 0:
        police.to_csv(PROCESSED_DIR / "police_stations.csv", index=False)
        print(f"   ✅ police_stations.csv 저장 ({len(police)} 건)")
    
    if len(convenience) > 0:
        convenience.to_csv(PROCESSED_DIR / "convenience_stores.csv", index=False)
        print(f"   ✅ convenience_stores.csv 저장 ({len(convenience)} 건)")
    
    if len(schools) > 0:
        schools.to_csv(PROCESSED_DIR / "schools.csv", index=False)
        print(f"   ✅ schools.csv 저장 ({len(schools)} 건)")
    
    if len(child_zones) > 0:
        child_zones.to_csv(PROCESSED_DIR / "child_protection_zones.csv", index=False)
        print(f"   ✅ child_protection_zones.csv 저장 ({len(child_zones)} 건)")
    
    if len(entertainment) > 0:
        entertainment.to_csv(PROCESSED_DIR / "entertainment_danger.csv", index=False)
        print(f"   ✅ entertainment_danger.csv 저장 ({len(entertainment)} 건)")
    
    # 도보 네트워크 저장
    if pedestrian:
        if 'nodes' in pedestrian and len(pedestrian['nodes']) > 0:
            pedestrian['nodes'].to_csv(PROCESSED_DIR / "pedestrian_nodes.csv", index=False)
            print(f"   ✅ pedestrian_nodes.csv 저장 ({len(pedestrian['nodes']):,} 건)")
        if 'links' in pedestrian and len(pedestrian['links']) > 0:
            pedestrian['links'].to_csv(PROCESSED_DIR / "pedestrian_links.csv", index=False)
            print(f"   ✅ pedestrian_links.csv 저장 ({len(pedestrian['links']):,} 건)")
    
    print("\n" + "=" * 60)
    print("✅ 전처리 완료!")
    print(f"   결과 저장 위치: {PROCESSED_DIR}")
    print("=" * 60)
    
    return {
        'streetlights': streetlights,
        'cctv': cctv,
        'police': police,
        'convenience': convenience,
        'schools': schools,
        'child_zones': child_zones,
        'entertainment': entertainment,
        'pedestrian': pedestrian
    }


if __name__ == "__main__":
    preprocess_all_data()

