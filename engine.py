
import pandas as pd
import numpy as np
import cv2, base64, ast, re, gcsfs
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import aiplatform

class DiagnosisEngine:
    def __init__(self, csv_path, endpoint_id, project_id="project-77db7a49-c886-49bb-8f6"):
        aiplatform.init(project=project_id, location="us-central1")
        self.endpoint = aiplatform.Endpoint(endpoint_id)
        self.fs = gcsfs.GCSFileSystem()
        self.df = pd.read_csv(csv_path)
        self.df['vec_arr'] = self.df['vector'].apply(ast.literal_eval)
        self.groups = {
            'TORS': ['TORS_RIGHT', 'TORS_LEFT'], 'ROT': ['ROT_RIGHT', 'ROT_LEFT'],
            'SAG': ['SAG_ANTERIOR', 'SAG_FLATTENING'], 'ASYM': ['ASYM_RIGHT', 'ASYM_LEFT']
        }

    def _preprocess(self, img):
        """이미지 전처리 로직 (prepare_image_bytes 기반)"""
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
        """전체 파이프라인 실행"""
        # 1. 이미지 복원 및 전처리
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_img = self._preprocess(img)
        
        # 2. 모델 예측
        _, encoded = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        img_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        pred = self.endpoint.predict(instances=[{"content": img_b64}]).predictions[0]
        
        # 3. 배타적 필터링 로직
        raw = {n: c for n, c in zip(pred.get('displayNames', []), pred.get('confidences', [])) if 'NONE' not in n}
        significant = []
        for g, keys in self.groups.items():
            cands = {k: raw.get(k, 0) for k in keys if k in raw}
            if cands:
                best = max(cands, key=cands.get)
                if cands[best] > 0.35: significant.append((best, cands[best]))
        for n, c in raw.items():
            if not any(n in v for v in self.groups.values()) and c > 0.35: significant.append((n, c))
            
        # 4. 코사인 유사도 검색
        order = ['TORS_RIGHT', 'TORS_LEFT', 'ROT_RIGHT', 'ROT_LEFT', 'SAG_ANTERIOR', 
                 'SAG_FLATTENING', 'ASYM_RIGHT', 'ASYM_LEFT', 'STATUS_SCOLIOSIS', 'STATUS_DEGENERATIVE']
        target_vec = np.array([dict(significant).get(k, 0.0) for k in order]).reshape(1, -1)
        sims = cosine_similarity(target_vec, np.stack(self.df['vec_arr'].values)).flatten()
        
        # 5. 결과 반환
        match_idx = np.argmax(sims)
        return {
            "diagnosis": significant,
            "best_match": self.df.iloc[match_idx],
            "similarity": sims[match_idx]
        }
