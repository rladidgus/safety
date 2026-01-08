"""
ML 기반 안전 가중치 학습 모듈 (생활인구 포함 버전)
- 피처 엔지니어링 강화 (밀도, 거리, 고립도)
- 생활인구 데이터 통합
- 복합 위험 지표
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML 패키지 미설치: pip install scikit-learn xgboost")

try:
    import networkx as nx
    from scipy.spatial import cKDTree
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


# ============================================
# 데이터 로드 함수들
# ============================================

def load_crime_data() -> pd.DataFrame:
    """경찰청 범죄 통계 로드 및 전처리"""
    filepath = DATA_DIR / "경찰청_범죄 발생 지역별 통계_20241231.csv"
    
    if not filepath.exists():
        print(f"❌ 파일 없음: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding='cp949')
    columns = df.columns.tolist()
    
    seoul_columns = [col for col in columns[2:] if '서울' in col or '서 울' in col]
    
    if not seoul_columns:
        return pd.DataFrame()
    
    crime_by_district = {}
    for col in seoul_columns:
        district = col.replace('서울 ', '').replace('서 울 ', '').strip()
        try:
            total_crimes = pd.to_numeric(df[col], errors='coerce').sum()
            crime_by_district[district] = int(total_crimes)
        except:
            continue
    
    result = pd.DataFrame([
        {'district': k, 'total_crimes': v}
        for k, v in crime_by_district.items()
    ])
    
    if len(result) > 0:
        max_crimes = result['total_crimes'].max()
        min_crimes = result['total_crimes'].min()
        result['danger_label'] = (result['total_crimes'] - min_crimes) / (max_crimes - min_crimes)
    
    print(f"✅ 서울시 범죄 데이터: {len(result)} 구")
    return result


def load_crime_time_data() -> Dict[str, float]:
    """
    범죄 발생 시간대별 데이터 로드
    반환: 시간대별 위험도 배율 (0~1 정규화)
    """
    # 시간대별 파일 찾기
    time_files = list(DATA_DIR.glob("범죄발생_시간_*.csv"))
    
    if not time_files:
        print("⚠️ 범죄 시간대 파일 없음")
        return get_default_time_danger()
    
    filepath = time_files[0]
    
    # 여러 인코딩 시도
    df = None
    for encoding in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']:
        try:
            df = pd.read_csv(filepath, encoding=encoding, header=[0, 1])
            break
        except:
            continue
    
    if df is None:
        print("⚠️ 범죄 시간대 파일 인코딩 오류")
        return get_default_time_danger()
    
    try:
        # 시간대별 범죄 건수
        time_slots = {
            '00:00-02:59': 0, '03:00-05:59': 0, '06:00-08:59': 0,
            '09:00-11:59': 0, '12:00-14:59': 0, '15:00-17:59': 0,
            '18:00-20:59': 0, '21:00-23:59': 0
        }
        
        for col in df.columns:
            col_str = str(col[1]) if isinstance(col, tuple) else str(col)
            for slot in time_slots.keys():
                if slot in col_str:
                    try:
                        val = pd.to_numeric(df.iloc[0][col], errors='coerce')
                        if pd.notna(val):
                            time_slots[slot] += val
                    except:
                        pass
        
        if sum(time_slots.values()) > 0:
            max_val = max(time_slots.values())
            min_val = min(time_slots.values())
            
            time_danger = {}
            for slot, val in time_slots.items():
                time_danger[slot] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            print(f"✅ 범죄 시간대 데이터 로드 완료")
            for slot, danger in sorted(time_danger.items()):
                print(f"   {slot}: {danger:.3f}")
            
            return time_danger
    except Exception as e:
        print(f"⚠️ 범죄 시간대 파일 처리 오류: {e}")
    
    return get_default_time_danger()


def load_crime_day_data() -> Dict[str, float]:
    """
    범죄 발생 요일별 데이터 로드
    반환: 요일별 위험도 배율 (0~1 정규화)
    """
    # 요일별 파일 찾기
    day_files = list(DATA_DIR.glob("범죄발생_요일_*.csv"))
    
    if not day_files:
        print("⚠️ 범죄 요일 파일 없음")
        return get_default_day_danger()
    
    filepath = day_files[0]
    
    # 여러 인코딩 시도
    df = None
    for encoding in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']:
        try:
            df = pd.read_csv(filepath, encoding=encoding, header=[0, 1])
            break
        except:
            continue
    
    if df is None:
        print("⚠️ 범죄 요일 파일 인코딩 오류")
        return get_default_day_danger()
    
    try:
        # 요일별 범죄 건수
        day_mapping = {
            '월': 'monday', '화': 'tuesday', '수': 'wednesday',
            '목': 'thursday', '금': 'friday', '토': 'saturday', '일': 'sunday'
        }
        
        day_counts = {day: 0 for day in day_mapping.values()}
        
        for col in df.columns:
            col_str = str(col[1]) if isinstance(col, tuple) else str(col)
            for kor, eng in day_mapping.items():
                if kor in col_str and '합계' not in col_str:
                    try:
                        val = pd.to_numeric(df.iloc[0][col], errors='coerce')
                        if pd.notna(val):
                            day_counts[eng] += val
                    except:
                        pass
        
        if sum(day_counts.values()) > 0:
            max_val = max(day_counts.values())
            min_val = min(day_counts.values())
            
            day_danger = {}
            for day, val in day_counts.items():
                day_danger[day] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            print(f"✅ 범죄 요일 데이터 로드 완료")
            for day, danger in day_danger.items():
                print(f"   {day}: {danger:.3f}")
            
            return day_danger
    except Exception as e:
        print(f"⚠️ 범죄 요일 파일 처리 오류: {e}")
    
    return get_default_day_danger()


def get_default_time_danger() -> Dict[str, float]:
    """기본 시간대별 위험도 (데이터 없을 때)"""
    return {
        '00:00-02:59': 0.9,   # 새벽: 매우 위험
        '03:00-05:59': 0.8,   # 새벽: 위험
        '06:00-08:59': 0.3,   # 아침: 안전
        '09:00-11:59': 0.2,   # 오전: 안전
        '12:00-14:59': 0.3,   # 점심: 안전
        '15:00-17:59': 0.4,   # 오후: 보통
        '18:00-20:59': 0.6,   # 저녁: 주의
        '21:00-23:59': 0.8,   # 밤: 위험
    }


def get_default_day_danger() -> Dict[str, float]:
    """기본 요일별 위험도 (데이터 없을 때)"""
    return {
        'monday': 0.4, 'tuesday': 0.4, 'wednesday': 0.4,
        'thursday': 0.5, 'friday': 0.7, 'saturday': 0.8, 'sunday': 0.6
    }


def get_time_danger_score(hour: int, time_danger: Dict[str, float]) -> float:
    """시간(0-23)에 해당하는 위험도 반환"""
    if 0 <= hour < 3:
        return time_danger.get('00:00-02:59', 0.9)
    elif 3 <= hour < 6:
        return time_danger.get('03:00-05:59', 0.8)
    elif 6 <= hour < 9:
        return time_danger.get('06:00-08:59', 0.3)
    elif 9 <= hour < 12:
        return time_danger.get('09:00-11:59', 0.2)
    elif 12 <= hour < 15:
        return time_danger.get('12:00-14:59', 0.3)
    elif 15 <= hour < 18:
        return time_danger.get('15:00-17:59', 0.4)
    elif 18 <= hour < 21:
        return time_danger.get('18:00-20:59', 0.6)
    else:
        return time_danger.get('21:00-23:59', 0.8)


def get_day_danger_score(day: int, day_danger: Dict[str, float]) -> float:
    """요일(0=월~6=일)에 해당하는 위험도 반환"""
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    return day_danger.get(days[day % 7], 0.5)


def load_streetlight_schedule() -> Dict[str, int]:
    """가로등 점소등 시간 로드"""
    filepath = DATA_DIR / "서울시 가로등 점소등 시간 현황.csv"
    
    if not filepath.exists():
        print("⚠️ 가로등 점소등 시간 파일 없음, 기본값 사용 (18:00~06:00)")
        return {'on_hour': 18, 'off_hour': 6}
    
    try:
        df = pd.read_csv(filepath, encoding='cp949')
        
        def parse_time(t):
            if pd.isna(t):
                return None
            return int(t) // 10000  # HHMMSS -> HH
        
        on_times = df['서울시 실제 점등시간(시분초)'].apply(parse_time).dropna()
        off_times = df['서울시 실제 소등시간(시분초)'].apply(parse_time).dropna()
        
        avg_on = int(on_times.mean()) if len(on_times) > 0 else 18
        avg_off = int(off_times.mean()) if len(off_times) > 0 else 6
        
        print(f"✅ 가로등 점소등 시간: 점등 {avg_on}시, 소등 {avg_off}시")
        return {'on_hour': avg_on, 'off_hour': avg_off}
        
    except Exception as e:
        print(f"⚠️ 가로등 점소등 시간 파일 오류: {e}")
        return {'on_hour': 18, 'off_hour': 6}


def is_streetlight_on(hour: int, schedule: Dict[str, int]) -> bool:
    """현재 시간에 가로등이 켜져 있는지 확인"""
    on_hour = schedule.get('on_hour', 18)
    off_hour = schedule.get('off_hour', 6)
    
    # 점등: 18시, 소등: 6시인 경우
    # 18~23시, 0~5시 = 가로등 ON
    if on_hour > off_hour:  # 저녁~아침 (일반적인 경우)
        return hour >= on_hour or hour < off_hour
    else:  # 예외 케이스
        return on_hour <= hour < off_hour


def load_population_data() -> pd.DataFrame:
    """생활인구 데이터 로드"""
    filepath = DATA_DIR / "all_months_monthly_avg_with_dong.csv"
    
    if not filepath.exists():
        print(f"⚠️ 생활인구 파일 없음: {filepath}")
        return pd.DataFrame()
    
    # 인코딩 시도
    for encoding in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            break
        except:
            continue
    else:
        print("❌ 생활인구 파일 인코딩 오류")
        return pd.DataFrame()
    
    # 컬럼명 정리
    df.columns = [col.strip() for col in df.columns]
    
    # 컬럼명 매핑 (다양한 형식 대응)
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if '동코드' in col or '행정동코드' in col or 'code' in col_lower:
            column_mapping[col] = 'dong_code'
        elif '동' in col and '인구' not in col:
            column_mapping[col] = 'dong_name'
        elif '생활인구' in col or '인구' in col or 'population' in col_lower:
            column_mapping[col] = 'population'
        elif 'month' in col_lower or '월' in col:
            column_mapping[col] = 'month'
    
    df = df.rename(columns=column_mapping)
    
    # 동별 평균 인구 계산 (월별 데이터가 있는 경우)
    if 'month' in df.columns and 'dong_code' in df.columns:
        df = df.groupby('dong_code').agg({
            'dong_name': 'first',
            'population': 'mean'
        }).reset_index()
    
    # 인구 정규화 (0~1)
    if 'population' in df.columns and len(df) > 0:
        pop_max = df['population'].max()
        pop_min = df['population'].min()
        df['population_normalized'] = (df['population'] - pop_min) / (pop_max - pop_min)
        
        print(f"✅ 생활인구 데이터: {len(df)} 동")
        print(f"   인구 범위: {pop_min:,.0f} ~ {pop_max:,.0f}")
    
    return df


def get_district_from_coords(lat: float, lon: float) -> str:
    """좌표로부터 구 이름 추정"""
    districts = {
        '강남구': (37.5172, 127.0473), '강동구': (37.5301, 127.1238),
        '강북구': (37.6396, 127.0255), '강서구': (37.5509, 126.8495),
        '관악구': (37.4784, 126.9516), '광진구': (37.5384, 127.0823),
        '구로구': (37.4954, 126.8874), '금천구': (37.4519, 126.9020),
        '노원구': (37.6542, 127.0568), '도봉구': (37.6688, 127.0471),
        '동대문구': (37.5744, 127.0400), '동작구': (37.5124, 126.9393),
        '마포구': (37.5663, 126.9014), '서대문구': (37.5791, 126.9368),
        '서초구': (37.4837, 127.0324), '성동구': (37.5633, 127.0371),
        '성북구': (37.5894, 127.0167), '송파구': (37.5145, 127.1059),
        '양천구': (37.5169, 126.8664), '영등포구': (37.5264, 126.8963),
        '용산구': (37.5324, 126.9907), '은평구': (37.6027, 126.9291),
        '종로구': (37.5735, 126.9790), '중구': (37.5641, 126.9979),
        '중랑구': (37.6063, 127.0925),
    }
    
    min_dist = float('inf')
    closest = '강남구'
    for district, (d_lat, d_lon) in districts.items():
        dist = ((lat - d_lat) ** 2 + (lon - d_lon) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = district
    return closest


def get_dong_code_from_coords(lat: float, lon: float, population_data: pd.DataFrame) -> str:
    """좌표로부터 가장 가까운 동 코드 반환"""
    if len(population_data) == 0 or 'dong_code' not in population_data.columns:
        return None
    
    # 동 코드에서 구 코드 추출하여 매칭 (간단한 구현)
    district = get_district_from_coords(lat, lon)
    
    # 해당 구의 동들 중 랜덤 선택 (실제로는 행정동 경계 데이터 필요)
    return population_data['dong_code'].iloc[np.random.randint(len(population_data))]


# ============================================
# 향상된 피처 엔지니어링 (생활인구 포함)
# ============================================

class EnhancedFeatureExtractor:
    """
    향상된 피처 추출기 (생활인구 포함)
    """
    
    def __init__(self, facilities: Dict[str, pd.DataFrame], 
                 population_data: pd.DataFrame = None):
        self.facilities = facilities
        self.population_data = population_data
        self.trees = {}
        self._build_spatial_indices()
        
        # 생활인구 딕셔너리 생성
        self.population_dict = {}
        if population_data is not None and len(population_data) > 0:
            if 'dong_code' in population_data.columns:
                self.population_dict = dict(zip(
                    population_data['dong_code'].astype(str),
                    population_data['population_normalized']
                ))
            if 'dong_name' in population_data.columns:
                name_dict = dict(zip(
                    population_data['dong_name'],
                    population_data['population_normalized']
                ))
                self.population_dict.update(name_dict)
    
    def _build_spatial_indices(self):
        """공간 인덱스 생성"""
        for key, df in self.facilities.items():
            if len(df) > 0 and 'latitude' in df.columns:
                coords = df[['latitude', 'longitude']].values
                self.trees[key] = cKDTree(coords)
                print(f"   ✅ {key}: {len(coords):,} 좌표 인덱싱")
            else:
                self.trees[key] = None
    
    def count_facilities(self, point: np.ndarray, facility_type: str, 
                         radius_m: float = 50) -> int:
        """반경 내 시설물 개수"""
        tree = self.trees.get(facility_type)
        if tree is None:
            return 0
        radius_deg = radius_m / 111000
        return len(tree.query_ball_point(point, radius_deg))
    
    def nearest_distance(self, point: np.ndarray, facility_type: str) -> float:
        """가장 가까운 시설물까지 거리 (미터)"""
        tree = self.trees.get(facility_type)
        if tree is None:
            return 1000.0
        
        dist_deg, _ = tree.query(point)
        return dist_deg * 111000
    
    def get_population_score(self, dong_code: str = None, dong_name: str = None) -> float:
        """동의 생활인구 점수 (0~1, 높을수록 인구 많음)"""
        if dong_code and str(dong_code) in self.population_dict:
            return self.population_dict[str(dong_code)]
        if dong_name and dong_name in self.population_dict:
            return self.population_dict[dong_name]
        return 0.5  # 기본값
    
    def extract_features(self, lat: float, lon: float, road_length: float = 100,
                         is_main_road: bool = False, dong_code: str = None,
                         dong_name: str = None, hour: int = None) -> Dict:
        """향상된 피처 추출 (생활인구 + 도로 피처 포함)"""
        point = np.array([lat, lon])
        
        # ===== 기본 피처: 개수 =====
        streetlight_count = self.count_facilities(point, 'streetlight', 50)
        cctv_count = self.count_facilities(point, 'cctv', 50)
        convenience_count = self.count_facilities(point, 'convenience', 100)
        entertainment_count = self.count_facilities(point, 'entertainment', 100)
        police_count = self.count_facilities(point, 'police', 500)
        school_count = self.count_facilities(point, 'school', 300)  # 학교 300m 반경
        child_zone_count = self.count_facilities(point, 'child_zone', 200)  # 어린이보호구역 200m
        
        # ===== 도보 네트워크 피처: 횡단보도/공원/터널 =====
        crosswalk_count = self.count_facilities(point, 'crosswalk', 100)  # 횡단보도 100m
        park_count = self.count_facilities(point, 'park', 100)  # 공원/녹지 100m
        tunnel_count = self.count_facilities(point, 'tunnel', 100)  # 터널 100m
        
        # ===== 밀도 피처 =====
        length_factor = max(road_length, 10) / 100
        streetlight_density = streetlight_count / length_factor
        cctv_density = cctv_count / length_factor
        convenience_density = convenience_count / length_factor
        entertainment_density = entertainment_count / length_factor
        
        # ===== 거리 피처 =====
        def normalize_distance(d, max_d=500):
            return max(0, 1 - d / max_d)
        
        streetlight_proximity = normalize_distance(self.nearest_distance(point, 'streetlight'), 100)
        cctv_proximity = normalize_distance(self.nearest_distance(point, 'cctv'), 200)
        convenience_proximity = normalize_distance(self.nearest_distance(point, 'convenience'), 300)
        police_proximity = normalize_distance(self.nearest_distance(point, 'police'), 1000)
        entertainment_proximity = normalize_distance(self.nearest_distance(point, 'entertainment'), 300)
        
        # ===== 고립도 =====
        no_streetlight = 1 if streetlight_count == 0 else 0
        no_cctv = 1 if cctv_count == 0 else 0
        no_convenience = 1 if convenience_count == 0 else 0
        no_police = 1 if police_count == 0 else 0
        
        isolation_score = (no_streetlight + no_cctv + no_convenience + no_police) / 4
        complete_isolation = 1 if (streetlight_count == 0 and cctv_count == 0 
                                   and convenience_count == 0 and police_count == 0) else 0
        
        # ===== 위험/안전 비율 =====
        safety_sum = streetlight_count + cctv_count * 2 + convenience_count + police_count * 3
        danger_sum = entertainment_count * 2
        
        if safety_sum > 0:
            danger_safety_ratio = danger_sum / (safety_sum + danger_sum)
        else:
            danger_safety_ratio = 1.0 if danger_sum > 0 else 0.5
        
        # ===== 도로/복합 피처 =====
        road_length_normalized = min(road_length / 500, 1)
        streetlight_coverage = min(streetlight_count / 3, 1)
        night_safety = min((streetlight_count * 0.3 + convenience_count * 0.5 + cctv_count * 0.2) / 3, 1)
        entertainment_danger = min(entertainment_count / 2, 1)
        
        # ===== 생활인구 피처 (NEW!) =====
        population_score = self.get_population_score(dong_code, dong_name)
        
        # 인구 적은 곳은 야간에 더 위험
        low_population = 1 if population_score < 0.3 else 0
        high_population = 1 if population_score > 0.7 else 0
        
        # 야간 고립 점수 (인구 적고 + 시설물 없음)
        night_isolation = isolation_score * (1 - population_score)
        
        return {
            # 기본 개수 피처
            'streetlight_count': streetlight_count,
            'cctv_count': cctv_count,
            'convenience_count': convenience_count,
            'entertainment_count': entertainment_count,
            'police_nearby': 1 if police_count > 0 else 0,
            
            # 밀도 피처
            'streetlight_density': streetlight_density,
            'cctv_density': cctv_density,
            'convenience_density': convenience_density,
            'entertainment_density': entertainment_density,
            
            # 거리 피처
            'streetlight_proximity': streetlight_proximity,
            'cctv_proximity': cctv_proximity,
            'convenience_proximity': convenience_proximity,
            'police_proximity': police_proximity,
            'entertainment_proximity': entertainment_proximity,
            
            # 고립도
            'isolation_score': isolation_score,
            'complete_isolation': complete_isolation,
            
            # 위험/안전 비율
            'danger_safety_ratio': danger_safety_ratio,
            
            # 도로 피처
            'road_length': road_length_normalized,
            'is_main_road': 1 if is_main_road else 0,
            
            # 복합 피처
            'streetlight_coverage': streetlight_coverage,
            'night_safety': night_safety,
            'entertainment_danger': entertainment_danger,
            
            # 생활인구 피처
            'population_score': population_score,
            'low_population': low_population,
            'high_population': high_population,
            'night_isolation': night_isolation,
            
            # 학교/어린이 보호구역 피처
            'school_nearby': 1 if school_count > 0 else 0,
            'child_zone_nearby': 1 if child_zone_count > 0 else 0,
            'safety_zone_score': min((school_count + child_zone_count) / 2, 1),
            
            # 도보 네트워크 피처 (횡단보도/공원/터널)
            'crosswalk_nearby': 1 if crosswalk_count > 0 else 0,
            'park_nearby': 1 if park_count > 0 else 0,  # 야간 위험 요소
            'tunnel_nearby': 1 if tunnel_count > 0 else 0,  # 위험 요소
            'road_safety_score': min(crosswalk_count / 2, 1) - 0.3 * (1 if park_count > 0 else 0) - 0.5 * (1 if tunnel_count > 0 else 0),
        }


# ============================================
# 향상된 ML 모델 (생활인구 포함)
# ============================================

class EnhancedSafetyMLModel:
    """향상된 안전 점수 예측 모델 (생활인구 포함)"""
    
    BASIC_FEATURES = [
        'streetlight_count', 'cctv_count', 'convenience_count',
        'entertainment_count', 'police_nearby'
    ]
    
    DENSITY_FEATURES = [
        'streetlight_density', 'cctv_density', 
        'convenience_density', 'entertainment_density'
    ]
    
    DISTANCE_FEATURES = [
        'streetlight_proximity', 'cctv_proximity', 'convenience_proximity',
        'police_proximity', 'entertainment_proximity'
    ]
    
    ISOLATION_FEATURES = [
        'isolation_score', 'complete_isolation', 'danger_safety_ratio'
    ]
    
    ROAD_FEATURES = [
        'road_length', 'is_main_road', 'streetlight_coverage',
        'night_safety', 'entertainment_danger'
    ]
    
    # 생활인구 피처
    POPULATION_FEATURES = [
        'population_score', 'low_population', 'high_population', 'night_isolation'
    ]
    
    # 시간/요일 피처
    TIME_FEATURES = [
        'hour_danger', 'day_danger', 'is_night', 'is_weekend'
    ]
    
    # 학교/어린이 보호구역 피처
    SAFETY_ZONE_FEATURES = [
        'school_nearby', 'child_zone_nearby', 'safety_zone_score'
    ]
    
    # 도보 네트워크 피처 (횡단보도/공원/터널)
    ROAD_NETWORK_FEATURES = [
        'crosswalk_nearby', 'park_nearby', 'tunnel_nearby', 'road_safety_score'
    ]
    
    # 가로등 점소등 피처
    STREETLIGHT_SCHEDULE_FEATURES = [
        'streetlight_on', 'streetlight_effective_count', 'streetlight_effective_proximity'
    ]
    
    def __init__(self, use_all_features: bool = True, use_population: bool = True,
                 use_time: bool = True, use_safety_zone: bool = True):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        
        self.feature_columns = self.BASIC_FEATURES.copy()
        
        if use_all_features:
            self.feature_columns += self.DENSITY_FEATURES
            self.feature_columns += self.DISTANCE_FEATURES
            self.feature_columns += self.ISOLATION_FEATURES
            self.feature_columns += self.ROAD_FEATURES
        
        if use_population:
            self.feature_columns += self.POPULATION_FEATURES
        
        if use_time:
            self.feature_columns += self.TIME_FEATURES
        
        if use_safety_zone:
            self.feature_columns += self.SAFETY_ZONE_FEATURES
            self.feature_columns += self.ROAD_NETWORK_FEATURES  # 도로 네트워크 피처
            self.feature_columns += self.STREETLIGHT_SCHEDULE_FEATURES  # 가로등 점소등 피처
        
        print(f"📊 사용할 피처 수: {len(self.feature_columns)}")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """피처 준비"""
        features = []
        for col in self.feature_columns:
            if col in df.columns:
                features.append(df[col].fillna(0).values)
            else:
                features.append(np.zeros(len(df)))
        return np.column_stack(features)
    
    def train(self, df: pd.DataFrame, target_col: str = 'danger_label') -> Dict:
        """모델 학습"""
        if not ML_AVAILABLE:
            raise ImportError("scikit-learn, xgboost 패키지를 설치해주세요.")
        
        print("\n" + "=" * 60)
        print("🤖 향상된 ML 모델 학습 (생활인구 포함)")
        print(f"   피처 수: {len(self.feature_columns)}")
        print("=" * 60)
        
        X = self.prepare_features(df)
        y = df[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        print(f"\n📊 모델 성능:")
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - R² (테스트): {r2:.4f}")
        print(f"   - R² (CV 평균): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.feature_importance = dict(zip(
            self.feature_columns, 
            self.model.feature_importances_
        ))
        
        print(f"\n📈 학습된 피처 중요도 (상위 10개):")
        sorted_importance = sorted(self.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
        for name, imp in sorted_importance:
            bar = '█' * int(imp * 50)
            print(f"   {name:25s}: {imp:.4f} {bar}")
        
        self.is_trained = True
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'cv_r2': cv_scores.mean()}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """위험도 예측"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, 1)
    
    def predict_single(self, **kwargs) -> float:
        """단일 예측"""
        df = pd.DataFrame([kwargs])
        return float(self.predict(df)[0])
    
    def save(self, filename: str = "enhanced_safety_model"):
        """모델 저장"""
        MODEL_DIR.mkdir(exist_ok=True)
        filepath = MODEL_DIR / f"{filename}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained
            }, f)
        
        print(f"\n💾 모델 저장: {filepath}")
    
    def load(self, filename: str = "enhanced_safety_model"):
        """모델 로드"""
        filepath = MODEL_DIR / f"{filename}.pkl"
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.feature_importance = data['feature_importance']
            self.is_trained = data['is_trained']
        
        print(f"✅ 모델 로드: {filepath}")


# 기존 호환용
SafetyMLModel = EnhancedSafetyMLModel


# ============================================
# 학습 데이터 생성
# ============================================

def load_facility_data() -> Dict[str, pd.DataFrame]:
    """시설물 데이터 로드"""
    facilities = {}
    files = {
        'streetlight': 'streetlights.csv',
        'cctv': 'cctv.csv',
        'police': 'police_stations.csv',
        'convenience': 'convenience_stores.csv',
        'entertainment': 'entertainment_danger.csv',
        'school': 'schools.csv',
        'child_zone': 'child_protection_zones.csv',
        'pedestrian_node': 'pedestrian_nodes.csv'  # 도보 네트워크 노드
    }
    
    for key, filename in files.items():
        filepath = PROCESSED_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            facilities[key] = df
        else:
            facilities[key] = pd.DataFrame()
    
    # 도보 링크 데이터도 로드 (횡단보도, 공원, 터널 분석용)
    links_path = PROCESSED_DIR / 'pedestrian_links.csv'
    if links_path.exists():
        links = pd.read_csv(links_path)
        facilities['pedestrian_links'] = links
        
        # 횡단보도/공원/터널 링크를 노드와 연결하여 좌표 추출
        nodes = facilities.get('pedestrian_node', pd.DataFrame())
        if len(nodes) > 0 and len(links) > 0:
            # 횡단보도 노드
            crosswalk_ids = links[links['crosswalk'] == 1]['start_node'].unique()
            facilities['crosswalk'] = nodes[nodes['node_id'].isin(crosswalk_ids)][['latitude', 'longitude']].copy()
            
            # 공원/녹지 노드 (야간 위험 요소)
            park_ids = links[links['park'] == 1]['start_node'].unique()
            facilities['park'] = nodes[nodes['node_id'].isin(park_ids)][['latitude', 'longitude']].copy()
            
            # 터널 노드 (위험 요소)
            tunnel_ids = links[links['tunnel'] == 1]['start_node'].unique()
            facilities['tunnel'] = nodes[nodes['node_id'].isin(tunnel_ids)][['latitude', 'longitude']].copy()
            
            print(f"   ✅ 횡단보도: {len(facilities['crosswalk']):,} 노드")
            print(f"   ✅ 공원/녹지: {len(facilities['park']):,} 노드")
            print(f"   ✅ 터널: {len(facilities['tunnel']):,} 노드")
    
    return facilities


def generate_training_data(crime_data: pd.DataFrame, 
                           facilities: Dict[str, pd.DataFrame],
                           population_data: pd.DataFrame = None,
                           time_danger: Dict[str, float] = None,
                           day_danger: Dict[str, float] = None,
                           streetlight_schedule: Dict[str, int] = None,
                           n_samples: int = 10000) -> pd.DataFrame:
    """향상된 피처로 학습 데이터 생성 (실제 도로 좌표 + 가로등 점소등 기반)"""
    
    print(f"\n📂 학습 데이터 생성 중 ({n_samples} 샘플)...")
    
    # 피처 추출기 생성 (생활인구 포함)
    extractor = EnhancedFeatureExtractor(facilities, population_data)
    
    # 시간/요일 위험도 (없으면 기본값)
    if time_danger is None:
        time_danger = get_default_time_danger()
    if day_danger is None:
        day_danger = get_default_day_danger()
    
    districts = crime_data['district'].tolist()
    danger_dict = dict(zip(crime_data['district'], crime_data['danger_label']))
    
    # 동 이름 리스트
    dong_names = []
    if population_data is not None and 'dong_name' in population_data.columns:
        dong_names = population_data['dong_name'].tolist()
    
    # 실제 도로 노드 좌표 사용 (핵심 변경!)
    pedestrian_nodes = facilities.get('pedestrian_node', pd.DataFrame())
    pedestrian_links = facilities.get('pedestrian_links', pd.DataFrame())
    
    if len(pedestrian_nodes) > 0:
        print(f"   ✅ 실제 도로 노드 좌표 사용: {len(pedestrian_nodes):,} 노드")
        # 노드에서 샘플링
        sample_indices = np.random.choice(len(pedestrian_nodes), min(n_samples, len(pedestrian_nodes)), replace=False)
        sampled_nodes = pedestrian_nodes.iloc[sample_indices].copy()
        
        # 노드에 해당하는 링크 정보 조인 (도로 길이 등)
        if len(pedestrian_links) > 0:
            node_link_info = pedestrian_links.groupby('start_node').agg({
                'length': 'mean',
                'crosswalk': 'max',
                'park': 'max',
                'tunnel': 'max'
            }).reset_index()
            sampled_nodes = sampled_nodes.merge(node_link_info, left_on='node_id', right_on='start_node', how='left')
    else:
        # 폴백: 랜덤 좌표 (도보 데이터 없을 경우)
        print("   ⚠️ 도보 네트워크 데이터 없음, 랜덤 좌표 사용")
        lat_range = (37.45, 37.70)
        lon_range = (126.80, 127.15)
        sampled_nodes = pd.DataFrame({
            'latitude': np.random.uniform(*lat_range, n_samples),
            'longitude': np.random.uniform(*lon_range, n_samples),
            'dong': None,
            'length': np.random.exponential(100, n_samples) + 10
        })
    
    np.random.seed(42)
    samples = []
    
    for idx, row in sampled_nodes.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        # 실제 도로 길이 사용 (있으면)
        road_length = row.get('length', 100)
        if pd.isna(road_length):
            road_length = 100
        road_length = min(road_length, 500)
        
        is_main_road = np.random.random() < 0.2
        
        # 동 이름 (노드에서 가져오거나 랜덤)
        dong_name = row.get('dong', None)
        if pd.isna(dong_name) and dong_names:
            dong_name = np.random.choice(dong_names)
        
        # 랜덤 시간/요일 (학습 데이터 다양성)
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)  # 0=월~6=일
        
        # 피처 추출
        features = extractor.extract_features(lat, lon, road_length, is_main_road, 
                                              dong_name=dong_name)
        
        # 시간/요일 피처 추가
        features['hour_danger'] = get_time_danger_score(hour, time_danger)
        features['day_danger'] = get_day_danger_score(day_of_week, day_danger)
        features['is_night'] = 1 if (hour < 6 or hour >= 21) else 0
        features['is_weekend'] = 1 if day_of_week >= 5 else 0
        
        # 가로등 ON/OFF 피처 추가
        if streetlight_schedule:
            sl_on = is_streetlight_on(hour, streetlight_schedule)
            features['streetlight_on'] = 1 if sl_on else 0
            # 가로등 꺼진 시간에는 가로등 효과 무효화
            if not sl_on:
                features['streetlight_effective_count'] = 0
                features['streetlight_effective_proximity'] = 0
            else:
                features['streetlight_effective_count'] = features['streetlight_count']
                features['streetlight_effective_proximity'] = features['streetlight_proximity']
        else:
            features['streetlight_on'] = 1
            features['streetlight_effective_count'] = features['streetlight_count']
            features['streetlight_effective_proximity'] = features['streetlight_proximity']
        
        district = get_district_from_coords(lat, lon)
        base_danger = danger_dict.get(district, 0.5)
        
        # 위험도 조정
        adjusted_danger = base_danger
        adjusted_danger += features['isolation_score'] * 0.2
        adjusted_danger += features['complete_isolation'] * 0.3
        
        # 가로등 효과 (점등 상태에 따라)
        adjusted_danger -= features['streetlight_effective_proximity'] * 0.1
        
        adjusted_danger -= features['cctv_proximity'] * 0.15
        adjusted_danger -= features['convenience_proximity'] * 0.1
        adjusted_danger -= features['police_proximity'] * 0.1
        adjusted_danger += features['entertainment_danger'] * 0.25
        
        # 생활인구 반영
        adjusted_danger += features['low_population'] * 0.15
        adjusted_danger -= features['high_population'] * 0.1
        adjusted_danger += features['night_isolation'] * 0.2
        
        # 시간/요일 반영
        adjusted_danger += features['hour_danger'] * 0.2  # 시간대 위험도 반영
        adjusted_danger += features['day_danger'] * 0.1   # 요일 위험도 반영
        adjusted_danger += features['is_night'] * 0.15    # 야간 추가 위험
        adjusted_danger += features['is_weekend'] * 0.05  # 주말 추가
        
        if is_main_road:
            adjusted_danger -= 0.1
        
        adjusted_danger = np.clip(adjusted_danger + np.random.normal(0, 0.05), 0, 1)
        
        sample = {
            'latitude': lat,
            'longitude': lon,
            'district': district,
            'hour': hour,
            'day_of_week': day_of_week,
            **features,
            'danger_label': adjusted_danger
        }
        samples.append(sample)
    
    df = pd.DataFrame(samples)
    print(f"✅ 학습 데이터 생성 완료: {len(df)} 샘플")
    
    return df


# ============================================
# 메인 실행
# ============================================

def train_and_save_model():
    """모델 학습 및 저장"""
    print("=" * 60)
    print("🚀 향상된 ML 모델 학습 (생활인구 + 시간/요일 포함)")
    print("=" * 60)
    
    # 1. 범죄 데이터 로드
    print("\n📂 범죄 데이터 로드...")
    crime_data = load_crime_data()
    
    if len(crime_data) == 0:
        print("❌ 범죄 데이터가 없습니다.")
        return None
    
    PROCESSED_DIR.mkdir(exist_ok=True)
    crime_data.to_csv(PROCESSED_DIR / "crime_by_district.csv", index=False, encoding='utf-8-sig')
    
    # 2. 생활인구 데이터 로드
    print("\n📂 생활인구 데이터 로드...")
    population_data = load_population_data()
    
    if len(population_data) > 0:
        population_data.to_csv(PROCESSED_DIR / "population_by_dong.csv", index=False, encoding='utf-8-sig')
    
    # 3. 범죄 시간/요일 데이터 로드
    print("\n📂 범죄 시간/요일 데이터 로드...")
    time_danger = load_crime_time_data()
    day_danger = load_crime_day_data()
    
    # 4. 가로등 점소등 시간 로드
    print("\n📂 가로등 점소등 시간 로드...")
    streetlight_schedule = load_streetlight_schedule()
    
    # 5. 시설물 데이터 로드
    print("\n📂 시설물 데이터 로드...")
    facilities = load_facility_data()
    
    for key, df in facilities.items():
        if len(df) > 0:
            print(f"   {key}: {len(df):,} 건")
    
    # 6. 학습 데이터 생성 (시간/요일 + 가로등 점소등 포함)
    training_data = generate_training_data(
        crime_data, facilities, population_data, 
        time_danger, day_danger, streetlight_schedule, n_samples=10000
    )
    
    # 6. 모델 학습
    use_population = len(population_data) > 0
    use_time = time_danger is not None
    use_safety_zone = 'school' in facilities and len(facilities.get('school', [])) > 0
    
    model = EnhancedSafetyMLModel(
        use_all_features=True, 
        use_population=use_population, 
        use_time=use_time,
        use_safety_zone=use_safety_zone
    )
    metrics = model.train(training_data)
    
    # 7. 모델 저장
    model.save("enhanced_safety_model")
    model.save("safety_ml_model")
    
    print("\n" + "=" * 60)
    print("✅ 향상된 ML 모델 학습 완료!")
    print(f"   R² 성능: {metrics['r2']:.4f}")
    if use_population:
        print("   📊 생활인구 피처 포함됨!")
    if use_time:
        print("   ⏰ 시간/요일 피처 포함됨!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    train_and_save_model()
