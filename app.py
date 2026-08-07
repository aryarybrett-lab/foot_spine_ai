import json
import streamlit as st
import base64
import gcsfs
from engine import DiagnosisEngine
import os
import requests

# --- 구글 클라우드 런 배포 완료된 URL 주소 ---
CLOUD_RUN_URL = "https://foot-spine-ai-952120235106.us-central1.run.app"

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="Star Docs: 족저압 기반 AI 스크리닝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 STARDOCs: AI 족저압 스크리닝 및 유사 환자 매칭 시스템")
st.markdown("족저압 결과지(JPG)를 업로드하시면, AI가 정밀 진단 소견을 도출하고 가장 유사한 임상 케이스의 X-ray를 비교해 드립니다.")

# --- 2. AI 엔진 및 GCS 파일시스템 로드 (캐싱을 통해 속도 최적화) ---
@st.cache_resource
def load_engine_and_gcs():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    csv_file = os.path.join(current_dir, 'vectorized_clinical_dataset.csv')
    
    # DiagnosisEngine 객체 생성
    eng = DiagnosisEngine(
        csv_path=csv_file,
        cloud_run_url=CLOUD_RUN_URL
    )
    
    # GCS 파일 시스템 인증 연동 (GCP_ADC_JSON 파싱)
    try:
        if hasattr(st, "secrets") and "GCP_ADC_JSON" in st.secrets:
            adc_raw = st.secrets["GCP_ADC_JSON"]
            adc_data = json.loads(adc_raw) if isinstance(adc_raw, str) else dict(adc_raw)
            fs_obj = gcsfs.GCSFileSystem(token=adc_data)
        elif os.environ.get("GCP_SA_KEY_JSON"):
            adc_data = json.loads(os.environ.get("GCP_SA_KEY_JSON"))
            fs_obj = gcsfs.GCSFileSystem(token=adc_data)
        else:
            fs_obj = gcsfs.GCSFileSystem()
    except Exception as e:
        print(f"GCS 파일 시스템 인증 경고: {e}")
        fs_obj = gcsfs.GCSFileSystem()
        
    return eng, fs_obj

with st.spinner("AI 엔진 및 클라우드 스토리지를 연결하는 중입니다..."):
    engine, fs = load_engine_and_gcs()

# --- 3. 사이드바: 파일 업로드 ---
st.sidebar.header("📁 환자 데이터 입력")
uploaded_file = st.sidebar.file_uploader("족저압 결과지 이미지 선택", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.sidebar.image(uploaded_file, caption="업로드된 족저압 결과지", use_container_width=True)
    
    if st.sidebar.button("🚀 AI 정밀 분석 시작", type="primary"):
        with st.spinner("🧠 Cloud Run TensorFlow 모델 연산 및 유사 환자 검색 중..."):
            img_bytes = uploaded_file.getvalue()
            result = engine.run_analysis_from_bytes(img_bytes, filename=uploaded_file.name)
            
        if result and "error" not in result:
            st.success("✨ 분석이 성공적으로 완료되었습니다!")
            
            # --- 결과 화면 레이아웃 (2개 컬럼) ---
            col1, col2 = st.columns([1, 1.2])
            
            # [컬럼 1] AI 정밀 진단 소견
            with col1:
                st.subheader("📋 AI 정밀 진단 소견")
                diagnosis_list = result.get('diagnosis', [])
                if diagnosis_list:
                    for diag, conf in diagnosis_list:
                        st.metric(label=diag, value=f"{conf:.1%}")
                else:
                    st.info("특이 소견 임계치(0.50)를 넘는 항목이 없습니다.")
                    
            # [컬럼 2] 최적 유사 환자 비교
            with col2:
                st.subheader("🔍 최적 유사 환자 비교")
                best_match = result.get('best_match')
                similarity = result.get('similarity', 0.0)
                
                # best_match 안전성 체크 (Series, Dict, DataFrame 고려)
                if best_match is not None and len(best_match) > 0:
                    # 항목 추출 함수 (Series 또는 Dict 처리)
                    def get_val(item, key):
                        if hasattr(item, 'get'):
                            return item.get(key, 'N/A')
                        elif key in item:
                            return item[key]
                        return 'N/A'

                    patient_id = get_val(best_match, 'foot_filename')
                    xray_ap = get_val(best_match, 'xray_ap_path')
                    xray_lat = get_val(best_match, 'xray_lat_path')

                    st.info(f"**매칭 환자 ID:** {patient_id} | **유사도:** {similarity:.2%}")
                    
                    # GCS에서 X-ray 이미지 바이트를 안전하게 읽어오는 함수
                   def get_img_b64(gs_path):
                        try:
                            if not gs_path or gs_path == 'N/A':
                                return None
                            
                            # 1. 파일명만 순수하게 추출 (예: 'P001_ap.jpg')
                            filename = str(gs_path).split('/')[-1]
                            
                            # 2. 실제 GCS 버킷/폴더 경로 결합
                            full_path = f"ai_foot_spine_image_bucket/x-ray_data/{filename}"
                            
                            with fs.open(full_path, 'rb') as f:
                                return base64.b64encode(f.read()).decode('utf-8')
                        except Exception as e:
                            print(f"GCS Image load error for {gs_path}: {e}")
                            return None
                    
                    ap_b64 = get_img_b64(xray_ap)
                    lat_b64 = get_img_b64(xray_lat)
                    
                    # X-ray 이미지 나란히 배치
                    xray_col1, xray_col2 = st.columns(2)
                    with xray_col1:
                        st.markdown("**[AP X-ray]**")
                        if ap_b64:
                            st.markdown(f'<img src="data:image/jpeg;base64,{ap_b64}" style="width:100%; border-radius:5px; border:1px solid #ddd;"/>', unsafe_allow_html=True)
                        else:
                            st.warning("AP 이미지를 불러올 수 없습니다.")
                            
                    with xray_col2:
                        st.markdown("**[Lateral X-ray]**")
                        if lat_b64:
                            st.markdown(f'<img src="data:image/jpeg;base64,{lat_b64}" style="width:100%; border-radius:5px; border:1px solid #ddd;"/>', unsafe_allow_html=True)
                        else:
                            st.warning("Lateral 이미지를 불러올 수 없습니다.")
                else:
                    st.warning("유사 환자 매칭 데이터를 찾을 수 없습니다.")
        else:
            st.error(f"분석에 실패했습니다: {result.get('error', '알 수 없는 오류')}")
else:
    st.markdown("---")
    st.info("👈 왼쪽 사이드바에서 족저압 결과지 이미지 파일을 업로드해 주세요.")
