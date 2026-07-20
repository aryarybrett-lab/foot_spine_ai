import streamlit as st
import base64
import gcsfs
from engine import DiagnosisEngine
import os

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Star Docs: 족저압 기반 AI 스크리닝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 Star Docs: AI 족저압 스크리닝 및 유사 환자 매칭 시스템")
st.markdown("족저압 결과지(JPG)를 업로드하시면, AI가 정밀 진단 소견을 도출하고 가장 유사한 임상 케이스의 X-ray를 비교해 드립니다.")

# --- 엔진 로드 (캐싱을 통해 속도 최적화) ---
@st.cache_resource
def load_engine():
    # 현재 스크립트가 있는 폴더 기준 상대 경로 설정 (코랩 & 클라우드 공용 호환)
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    csv_file = os.path.join(current_dir, 'vectorized_clinical_dataset.csv')
    
    return DiagnosisEngine(
        csv_path=csv_file,
        endpoint_id='6036926316664586240'
    )

with st.spinner("AI 엔진을 불러오는 중입니다..."):
    engine = load_engine()
    fs = gcsfs.GCSFileSystem()

# --- 사이드바: 파일 업로드 ---
st.sidebar.header("📁 환자 데이터 입력")
uploaded_file = st.sidebar.file_uploader("족저압 결과지 이미지 선택", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 업로드한 이미지 미리보기
    st.sidebar.image(uploaded_file, caption="업로드된 족저압 결과지", use_container_width=True)
    
    if st.sidebar.button("🚀 AI 정밀 분석 시작", type="primary"):
        with st.spinner("🧠 족저압 패턴 분석 및 유사 환자 검색 중..."):
            # 엔진 실행
            img_bytes = uploaded_file.getvalue()
            result = engine.run_analysis_from_bytes(img_bytes, filename=uploaded_file.name)
            
        st.success("✨ 분석이 완료되었습니다!")
        
        # --- 결과 화면 레이아웃 (2개 컬럼) ---
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("📋 AI 정밀 진단 소견")
            diagnosis_list = result['diagnosis']
            if diagnosis_list:
                for diag, conf in diagnosis_list:
                    st.metric(label=diag, value=f"{conf:.1%}")
            else:
                st.info("특이 소견 임계치(0.35)를 넘는 항목이 없습니다.")
                
        with col2:
            st.subheader("🔍 최적 유사 환자 비교")
            best_match = result['best_match']
            similarity = result['similarity']
            
            st.info(f"**매칭 환자 ID:** {best_match['foot_filename']} | **유사도:** {similarity:.2%}")
            
            # GCS에서 X-ray 이미지 바이트를 읽어와서 Base64로 변환 후 렌더링
            def get_img_b64(gs_path):
                try:
                    with fs.open(gs_path, 'rb') as f:
                        return base64.b64encode(f.read()).decode('utf-8')
                except:
                    return None
            
            ap_b64 = get_img_b64(best_match['xray_ap_path'])
            lat_b64 = get_img_b64(best_match['xray_lat_path'])
            
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
    st.markdown("---")
    st.info("👈 왼쪽 사이드바에서 족저압 결과지 이미지 파일을 업로드해 주세요.")