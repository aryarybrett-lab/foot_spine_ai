import json
import pandas as pd
import numpy as np
import cv2, ast, gcsfs
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

class DiagnosisEngine:
    def __init__(self, csv_path, cloud_run_url="https://foot-spine-ai-952120235106.us-central1.run.app"):
        self.cloud_run_url = cloud_run_url.rstrip('/')
        
        # GCS 파일 시스템 연동 (X-ray 이미지 로드용)
        if hasattr(st, "secrets") and "GCP_ADC_JSON" in st.secrets:
            try:
                adc_raw = st.secrets["GCP_ADC_JSON"]
                adc_data = json.loads(adc_raw) if isinstance(adc_raw, str) else dict(adc_raw)
                self.fs = gcsfs.GCSFileSystem(token=adc_data)
            except Exception as e:
                print(f"GCSFS 인증 토큰 읽기 경고: {e}")
                self.fs = gcsfs.GCSFileSystem()
        else:
            self.fs = gcsfs.GCSFileSystem()

        # 임상 벡터 데이터셋 CSV 로드 및 가공
        self.df = pd.read_csv(csv_path)
        self.df['vec_arr'] = self.df['vector'].apply(ast.literal_eval)
        
        # 질환 그룹화 구조
        self.groups = {
            'TORS': ['TORS_RIGHT', 'TORS_LEFT'], 
            'ROT': ['ROT_RIGHT', 'ROT_LEFT'],
            'SAG': ['SAG_ANTERIOR', 'SAG_FLATTENING'], 
            'ASYM': ['ASYM_RIGHT', 'ASYM_LEFT']
        }
        
        # 10차원 벡터 정렬 순서 정의
        self.order = [
            'TORS_RIGHT', 'TORS_LEFT', 'ROT_RIGHT', 'ROT_LEFT', 'SAG_ANTERIOR', 
            'SAG_FLATTENING', 'ASYM_RIGHT', 'ASYM_LEFT', 'STATUS_SCOLIOSIS', 'STATUS_DEGENERATIVE'
        ]

    def _preprocess(self, img):
        """족저압 결과지 영역 크롭 및 224x224 리사이징 전처리"""
        img_resized = cv2.resize(img, (602, 851))
        static_area = img_resized[138:138+175, 28:28+270]
        dynamic_area = img_resized[489:489+175, 28:28+270]
        stacked = np.vstack((static_area, dynamic_area))
        
        size = 224
        h, w = stacked.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(stacked, (int(w*scale), int(h*scale)))
        final = np.zeros((size, size, 3), dtype=np.uint8)
        final[(size-resized.shape[0])//2:(size-resized.shape[0])//2+resized.shape[0], 
              (size-resized.shape[1])//2:(size-resized.shape[1])//2+resized.shape[1]] = resized
        return final

    def run_analysis_from_bytes(self, img_bytes, filename="U_00U_00.jpg"):
        """Cloud Run API 호출 후 10차원 벡터 파싱 및 코사인 유사도 환자 매칭"""
        try:
            # 1. 이미지 읽기 및 전처리
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            processed_img = self._preprocess(img)
            
            # 2. 전처리된 이미지를 바이너리(JPEG)로 인코딩
            _, encoded = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            processed_bytes = encoded.tobytes()

            # 3. Cloud Run API로 이미지 전송 (`/predict` 엔드포인트)
            predict_url = f"{self.cloud_run_url}/predict"
            files = {"file": (filename, processed_bytes, "image/jpeg")}

            response = requests.post(predict_url, files=files, timeout=15)

            if response.status_code != 200:
                return {"error": f"Cloud Run API 오류 (코드: {response.status_code}): {response.text[:100]}"}

            # JSON 응답 여부 체크
            try:
                res_json = response.json()
            except Exception:
                return {"error": f"Cloud Run 응답이 JSON 포맷이 아닙니다: {response.text[:100]}"}
            
            # 4. Cloud Run 응답 데이터 파싱
            # API 응답 구조: {"predictions_vector": [...], "diagnosis_scores": {...}}
            diagnosis_scores = res_json.get("diagnosis_scores", {})
            predictions_vector = res_json.get("predictions_vector", [])

            # 백엔드 API에서 라벨 맵(diagnosis_scores)이 오는 경우 직접 정제
            raw = {n: c for n, c in diagnosis_scores.items() if 'NONE' not in n}
            significant = []
            
            for g, keys in self.groups.items():
                cands = {k: raw.get(k, 0) for k in keys if k in raw}
                if cands:
                    best = max(cands, key=cands.get)
                    if cands[best] > 0.50: 
                        significant.append((best, cands[best]))
                        
            for n, c in raw.items():
                if not any(n in v for v in self.groups.values()) and c > 0.50: 
                    significant.append((n, c))

            # 5. 타겟 10차원 벡터 생성 및 코사인 유사도 매칭
            if predictions_vector and len(predictions_vector) == 10:
                target_vec = np.array(predictions_vector).reshape(1, -1)
            else:
                target_vec = np.array([dict(significant).get(k, 0.0) for k in self.order]).reshape(1, -1)
                
            sims = cosine_similarity(target_vec, np.stack(self.df['vec_arr'].values)).flatten()
            match_idx = np.argmax(sims)
            
            return {
                "diagnosis": significant,
                "best_match": self.df.iloc[match_idx].to_dict(),,
                "similarity": sims[match_idx]
            }

        except Exception as e:
            return {"error": f"분석 프로세스 오류: {str(e)}"}
