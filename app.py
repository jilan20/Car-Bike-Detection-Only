import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import tensorflow as tf
import io
import time

ULTRALYTICS_AVAILABLE = False
try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except:
    pass

st.set_page_config(page_title="Car & Bike Detection AI", page_icon="🚗", layout="wide")

# CSS ------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Reset & Base */
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    /* FORCE HIDE ALL STREAMLIT MENU */
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    #MainMenu,
    header,
    footer,
    button[title="View app menu"],
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137,
    .stDeployButton,
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] > div:first-child,
    .css-1dp5vir,
    .css-1v0mbdj,
    .stToolbar {
      visibility: hidden !important;
      display: none !important;
      opacity: 0 !important;
      pointer-events: none !important;
      width: 0 !important;
      height: 0 !important;
    }
    
    /* App Background */
    .stApp {
      background: linear-gradient(135deg, #0f0b1a 0%, #1a1330 50%, #251a40 100%);
      color: #e8e4f3;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Section */
    .hero-title {
      font-size: 64px;
      font-weight: 800;
      text-align: center;
      margin: 32px 0 16px 0;
      background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 50%, #e0b3ff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: -1.5px;
      line-height: 1.1;
    }
    
    .hero-sub { 
      text-align: center; 
      color: rgba(255, 255, 255, 0.7);
      margin-bottom: 48px;
      font-size: 18px;
      font-weight: 400;
      line-height: 1.7;
      max-width: 700px;
      margin-left: auto;
      margin-right: auto;
    }

    /* Card Styling */
    .card {
      background: rgba(255, 255, 255, 0.03);
      backdrop-filter: blur(20px);
      border-radius: 20px;
      padding: 40px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Feature Cards */
    .feature {
      background: rgba(255, 255, 255, 0.04);
      backdrop-filter: blur(10px);
      border-radius: 16px;
      padding: 36px 28px;
      text-align: center;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      border: 1px solid rgba(255, 255, 255, 0.1);
      height: 100%;
      position: relative;
      overflow: hidden;
    }
    
    .feature::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, #a78bfa, #c4b5fd);
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    
    .feature:hover {
      transform: translateY(-8px);
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(167, 139, 250, 0.3);
      box-shadow: 0 12px 40px rgba(167, 139, 250, 0.2);
    }
    
    .feature:hover::before {
      opacity: 1;
    }
    
    .feature img {
      filter: drop-shadow(0 4px 12px rgba(167, 139, 250, 0.3));
    }
    
    .feature h3 {
      font-size: 20px;
      font-weight: 700;
      margin: 20px 0 12px 0;
      color: #e0b3ff;
    }

    /* Buttons */
    .stButton > button {
      background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
      color: white;
      border-radius: 12px;
      padding: 16px 36px;
      font-weight: 600;
      font-size: 16px;
      border: none;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
      cursor: pointer;
      letter-spacing: 0.3px;
    }
    
    .stButton > button:hover { 
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(124, 58, 237, 0.5);
      background: linear-gradient(135deg, #6d28d9 0%, #9333ea 100%);
    }
    
    .stButton > button:active {
      transform: translateY(0);
    }

    /* Typography */
    .muted { 
      color: rgba(255, 255, 255, 0.65);
      font-size: 15px;
      line-height: 1.7;
    }

    h3 {
      color: #e0b3ff;
      font-weight: 700;
      font-size: 28px;
      margin-bottom: 20px;
      letter-spacing: -0.5px;
    }
    
    h4 {
      color: rgba(255, 255, 255, 0.9);
      font-weight: 600;
      font-size: 18px;
    }

    /* Settings Box */
    .settings-box {
      background: rgba(255, 255, 255, 0.04);
      backdrop-filter: blur(10px);
      padding: 28px;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Stats Cards */
    .stat-card {
      background: linear-gradient(135deg, rgba(167, 139, 250, 0.1), rgba(124, 58, 237, 0.05));
      border-radius: 16px;
      padding: 28px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      text-align: center;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      overflow: hidden;
    }
    
    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: linear-gradient(90deg, #a78bfa, #c4b5fd);
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    
    .stat-card:hover {
      transform: translateY(-6px);
      border: 1px solid rgba(167, 139, 250, 0.3);
      box-shadow: 0 12px 40px rgba(124, 58, 237, 0.2);
    }
    
    .stat-card:hover::before {
      opacity: 1;
    }
    
    .stat-number {
      font-size: 52px;
      font-weight: 800;
      background: linear-gradient(135deg, #a78bfa 0%, #e0b3ff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin: 12px 0;
      letter-spacing: -1px;
    }
    
    .stat-label {
      color: rgba(255, 255, 255, 0.7);
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    
    .stat-sublabel {
      color: rgba(255, 255, 255, 0.5);
      font-size: 13px;
      margin-top: 6px;
    }

    /* Progress Bar */
    .accuracy-bar {
      background: rgba(255, 255, 255, 0.03);
      border-radius: 12px;
      padding: 20px;
      margin: 20px 0;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .progress-track {
      background: rgba(255, 255, 255, 0.05);
      border-radius: 12px;
      height: 28px;
      overflow: hidden;
      margin-top: 12px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .progress-fill {
      background: linear-gradient(90deg, #7c3aed 0%, #a78bfa 50%, #c4b5fd 100%);
      height: 100%;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding-right: 16px;
      color: white;
      font-weight: 700;
      font-size: 14px;
      transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: inset 0 2px 8px rgba(255, 255, 255, 0.2);
    }

    /* Detail Table */
    .detail-table {
      background: rgba(255, 255, 255, 0.03);
      border-radius: 16px;
      padding: 24px;
      margin: 24px 0;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .table-header {
      display: grid;
      grid-template-columns: 1fr 1fr 1.5fr 1.5fr;
      gap: 16px;
      padding: 16px 20px;
      background: rgba(167, 139, 250, 0.1);
      border-radius: 10px;
      margin-bottom: 12px;
      font-weight: 700;
      color: #e0b3ff;
      font-size: 14px;
      letter-spacing: 0.5px;
    }
    
    .table-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1.5fr 1.5fr;
      gap: 16px;
      padding: 18px 20px;
      background: rgba(255, 255, 255, 0.03);
      border-radius: 10px;
      margin-bottom: 8px;
      align-items: center;
      transition: all 0.2s ease;
      border: 1px solid transparent;
    }
    
    .table-row:hover {
      background: rgba(167, 139, 250, 0.08);
      transform: translateX(4px);
      border: 1px solid rgba(167, 139, 250, 0.2);
    }
    
    .class-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 13px;
    }
    
    .badge-car {
      background: rgba(124, 58, 237, 0.2);
      color: #c4b5fd;
      border: 1px solid rgba(124, 58, 237, 0.3);
    }
    
    .badge-bike {
      background: rgba(167, 139, 250, 0.2);
      color: #e0b3ff;
      border: 1px solid rgba(167, 139, 250, 0.3);
    }

    /* Classification Card */
    .classification-card {
      background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(124, 58, 237, 0.1));
      border-radius: 16px;
      padding: 24px;
      margin: 16px 0;
      border: 1px solid rgba(167, 139, 250, 0.2);
      position: relative;
      overflow: hidden;
    }
    
    .classification-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, #a78bfa, #c4b5fd);
    }

    /* Classification Result */
    .class-result {
      display: flex;
      align-items: center;
      gap: 20px;
      padding: 20px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 12px;
      margin: 12px 0;
      transition: all 0.3s ease;
    }
    
    .class-result:hover {
      background: rgba(255, 255, 255, 0.08);
      transform: translateX(4px);
    }
    
    .class-icon {
      font-size: 48px;
      width: 70px;
      height: 70px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(167, 139, 250, 0.2);
      border-radius: 12px;
      border: 2px solid rgba(167, 139, 250, 0.3);
    }
    
    .class-info {
      flex: 1;
    }
    
    .class-name {
      font-size: 24px;
      font-weight: 700;
      color: #e0b3ff;
      margin-bottom: 8px;
    }
    
    .class-conf {
      font-size: 16px;
      color: rgba(255, 255, 255, 0.7);
    }

    /* Confidence Meter */
    .conf-meter {
      margin-top: 16px;
    }
    
    .conf-bar {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      height: 24px;
      overflow: hidden;
      position: relative;
    }
    
    .conf-fill {
      height: 100%;
      background: linear-gradient(90deg, #7c3aed, #a78bfa);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding-right: 12px;
      color: white;
      font-weight: 600;
      font-size: 12px;
      transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Welcome Banner */
    .welcome-banner {
      background: linear-gradient(135deg, rgba(167, 139, 250, 0.12), rgba(124, 58, 237, 0.08));
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 32px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      text-align: center;
    }
    
    .welcome-banner h4 {
      margin: 0;
      color: #e0b3ff;
      font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
      width: 10px;
    }
    
    ::-webkit-scrollbar-track {
      background: rgba(255, 255, 255, 0.03);
    }
    
    ::-webkit-scrollbar-thumb {
      background: rgba(167, 139, 250, 0.3);
      border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
      background: rgba(167, 139, 250, 0.5);
    }
    
    /* Animation */
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .card {
      animation: fadeIn 0.6s ease-out;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      color: #e8e4f3;
      font-size: 16px;
      padding: 14px 16px;
      transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
      border: 1px solid rgba(167, 139, 250, 0.5);
      box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1);
      outline: none;
    }
    
    /* File Uploader */
    .stFileUploader > div {
      background: rgba(255, 255, 255, 0.03);
      border: 2px dashed rgba(167, 139, 250, 0.3);
      border-radius: 16px;
      padding: 40px;
      transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
      border-color: rgba(167, 139, 250, 0.5);
      background: rgba(255, 255, 255, 0.05);
    }

    .block-container {
        padding-top: 2rem !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if "page" not in st.session_state:
    st.session_state.page = 0
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def go_next():
    if st.session_state.page < 3:
        st.session_state.page += 1

def go_prev():
    if st.session_state.page > 0:
        st.session_state.page -= 1

# Model Loading
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# Classification function
def classify_crop(crop_img, classifier_model):
    """Classify cropped vehicle image as car or bike"""
    try:
        # Resize to model input size (adjust based on your model)
        img_resized = crop_img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = classifier_model.predict(img_array, verbose=0)
        
        # Assuming binary classification: 0=bike, 1=car
        if prediction[0][0] > 0.5:
            return "car", float(prediction[0][0])
        else:
            return "bike", float(1 - prediction[0][0])
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return "unknown", 0.0

# Header
st.markdown("<div class='hero-title'>Car & Bike Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Platform deteksi serta klasifikasi kendaraan berbasis AI menggunakan teknologi Computer Vision dan Deep Learning</div>", unsafe_allow_html=True)

content_col = st.container()

# Page 0 - Landing
if st.session_state.page == 0:
    with content_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,1,1], gap="large")
        
        with c1:
            st.markdown("""
                <div class='feature'>
                    <img src='https://cdn-icons-png.flaticon.com/512/9574/9574268.png' 
                     style='width:80px; height:80px; object-fit:contain;'/>
                    <h3>Deteksi Akurat</h3>
                    <p class='muted'>Identifikasi dan lokalisasi mobil & motor pada gambar dengan bounding box presisi tinggi</p>
                </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown("""
                <div class='feature'>
                    <img src='https://cdn-icons-png.flaticon.com/512/8618/8618881.png' 
                     style='width:80px; height:80px; object-fit:contain;'/>
                    <h3>AI Powered</h3>
                    <p class='muted'>Memanfaatkan YOLOv8n untuk inference cepat dan akurasi tinggi dalam deteksi real-time</p>
                </div>
            """, unsafe_allow_html=True)
        
        with c3:
            st.markdown("""
                <div class='feature'>
                    <img src='https://cdn-icons-png.flaticon.com/512/2920/2920234.png' 
                     style='width:80px; height:80px; object-fit:contain; filter: brightness(0) invert(1);'/>
                    <h3>Klasifikasi Cerdas</h3>
                    <p class='muted'>Klasifikasi otomatis setiap kendaraan terdeteksi dengan deep learning classifier</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='text-align:center;margin-top:16px'><p class='muted' style='font-size:14px'>🚀 Gratis · ⚡ Tanpa Registrasi · 📊 Hasil Instan · 🔒 Privasi Terjamin</p></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Mulai Deteksi", use_container_width=True):
                go_next()

# Page 1 - How it works
elif st.session_state.page == 1:
    with content_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")
        
        with left:
            st.markdown("<h3 style='color:#fff; font-size:22px; margin-bottom:16px; font-weight:600;'>Bagaimana Sistem Bekerja?</h3>", unsafe_allow_html=True)
            
            st.markdown("<div style='background:rgba(255,255,255,0.03); padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
            
            st.markdown("""
                <div style='margin-bottom:18px; padding-left:12px; border-left:3px solid #b794f6;'>
                    <h4 style='color:#b794f6; font-size:16px; margin-bottom:6px; font-weight:600;'>📤 Input Processing</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Gambar diunggah dan dinormalisasi untuk memastikan format dan ukuran sesuai standar model
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='margin-bottom:18px; padding-left:12px; border-left:3px solid #67c6f4;'>
                    <h4 style='color:#67c6f4; font-size:16px; margin-bottom:6px; font-weight:600;'>🎯 AI Detection</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Model YOLOv8n menganalisis gambar dan memprediksi lokasi setiap kendaraan dengan akurasi tinggi
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='margin-bottom:18px; padding-left:12px; border-left:3px solid #9f7aea;'>
                    <h4 style='color:#9f7aea; font-size:16px; margin-bottom:6px; font-weight:600;'>🔍 Deep Classification</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Setiap kendaraan terdeteksi diklasifikasi menggunakan CNN classifier untuk menentukan jenis kendaraan
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='padding-left:12px; border-left:3px solid #48bb78;'>
                    <h4 style='color:#48bb78; font-size:16px; margin-bottom:6px; font-weight:600;'>📊 Visualization</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Hasil ditampilkan dengan bounding box berwarna, label klasifikasi, dan statistik lengkap untuk analisis
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with right:
            st.markdown("<h3 style='color:#fff; font-size:20px; margin-bottom:20px; font-weight:600;'>Keunggulan Sistem</h3>", unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:rgba(183,148,246,0.1); padding:20px; border-radius:10px; margin-bottom:16px; 
                     border-left:3px solid #b794f6;'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:36px; height:36px; background:rgba(183,148,246,0.2); border-radius:8px; 
                             display:flex; align-items:center; justify-content:center; margin-right:14px; flex-shrink:0;'>
                            ⚡
                        </div>
                        <div>
                            <h4 style='color:#fff; font-size:16px; margin:0 0 6px 0; font-weight:600;'>Kecepatan Tinggi</h4>
                            <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                                Proses deteksi dan klasifikasi hanya membutuhkan beberapa detik dengan inference time yang optimal
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:rgba(103,198,244,0.1); padding:20px; border-radius:10px; margin-bottom:16px; 
                     border-left:3px solid #67c6f4;'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:36px; height:36px; background:rgba(103,198,244,0.2); border-radius:8px; 
                             display:flex; align-items:center; justify-content:center; margin-right:14px; flex-shrink:0;'>
                            🎯
                        </div>
                        <div>
                            <h4 style='color:#fff; font-size:16px; margin:0 0 6px 0; font-weight:600;'>Akurasi Presisi</h4>
                            <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                                Dual-model architecture dengan YOLO untuk deteksi dan CNN untuk klasifikasi memberikan hasil yang akurat
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:
