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

#Session state
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

#Model 
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# Classification function - PERBAIKAN
def classify_crop(crop_img, classifier_model):
    """Classify cropped vehicle image as car or bike"""
    try:
        # Model mengharapkan input 9216 = 96x96x1 (grayscale)
        img_resized = crop_img.resize((96, 96))
        img_gray = img_resized.convert('L')  # Convert to grayscale
        img_array = np.array(img_gray) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Shape: (96, 96, 1)
        img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 96, 96, 1)
        
        # Flatten jika model butuh flatten input
        # img_array = img_array.reshape(1, -1)  # Shape: (1, 9216)
        
        # Predict
        prediction = classifier_model.predict(img_array, verbose=0)
        
        # Binary classification: 0=bike, 1=car
        if prediction[0][0] > 0.5:
            return "car", float(prediction[0][0])
        else:
            return "bike", float(1 - prediction[0][0])
            
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return "unknown", 0.0

#Header
st.markdown("<div class='hero-title'>Car & Bike Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Platform deteksi serta klasifikasi kendaraan berbasis AI menggunakan teknologi Computer Vision dan Deep Learning</div>", unsafe_allow_html=True)

content_col = st.container()

#Halaman 0----------------------------------------------
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
        
        st.markdown("<div style='text-align:center;margin-top:16px'><p class='muted' style='font-size:14px'>Gratis · Tanpa Registrasi · Hasil Instan · Privasi Terjamin</p></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Mulai Deteksi", use_container_width=True):
                go_next()

#Halaman 1 --------------------------------------------------------
elif st.session_state.page == 1:
    with content_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")
        
        with left:
            st.markdown("<h3 style='color:#e0b3ff; font-size:24px; margin-bottom:24px; font-weight:700;'>Bagaimana Sistem Bekerja?</h3>", unsafe_allow_html=True)
            
            st.markdown("<div style='background:rgba(255,255,255,0.02); padding:24px; border-radius:12px; border:1px solid rgba(255,255,255,0.05)'>", unsafe_allow_html=True)
            
            st.markdown("""
                <div style='margin-bottom:20px; padding:18px; background:rgba(183,148,246,0.08); border-radius:10px; border-left:4px solid #b794f6;'>
                    <h4 style='color:#d4c5f9; font-size:16px; margin-bottom:8px; font-weight:700; letter-spacing:0.3px;'>INPUT PROCESSING</h4>
                    <p style='color:rgba(255,255,255,0.65); font-size:14px; margin:0; line-height:1.6;'>
                        Gambar diunggah dan dinormalisasi untuk memastikan format dan ukuran sesuai standar model AI
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='margin-bottom:20px; padding:18px; background:rgba(103,198,244,0.08); border-radius:10px; border-left:4px solid #67c6f4;'>
                    <h4 style='color:#9ed7f5; font-size:16px; margin-bottom:8px; font-weight:700; letter-spacing:0.3px;'>OBJECT DETECTION</h4>
                    <p style='color:rgba(255,255,255,0.65); font-size:14px; margin:0; line-height:1.6;'>
                        Model YOLOv8n menganalisis gambar dan memprediksi lokasi setiap kendaraan dengan akurasi tinggi menggunakan bounding box
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='margin-bottom:20px; padding:18px; background:rgba(159,122,234,0.08); border-radius:10px; border-left:4px solid #9f7aea;'>
                    <h4 style='color:#c4a7f4; font-size:16px; margin-bottom:8px; font-weight:700; letter-spacing:0.3px;'>DEEP CLASSIFICATION</h4>
                    <p style='color:rgba(255,255,255,0.65); font-size:14px; margin:0; line-height:1.6;'>
                        Setiap kendaraan terdeteksi diklasifikasi menggunakan CNN classifier untuk menentukan jenis kendaraan secara presisi
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='padding:18px; background:rgba(72,187,120,0.08); border-radius:10px; border-left:4px solid #48bb78;'>
                    <h4 style='color:#7ed6a3; font-size:16px; margin-bottom:8px; font-weight:700; letter-spacing:0.3px;'>VISUALIZATION & ANALYTICS</h4>
                    <p style='color:rgba(255,255,255,0.65); font-size:14px; margin:0; line-height:1.6;'>
                        Hasil ditampilkan dengan bounding box berwarna, label klasifikasi, dan statistik lengkap untuk analisis mendalam
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with right:
            st.markdown("<h3 style='color:#e0b3ff; font-size:24px; margin-bottom:24px; font-weight:700;'>Keunggulan Sistem</h3>", unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:rgba(183,148,246,0.08); padding:20px; border-radius:12px; margin-bottom:16px; 
                     border:1px solid rgba(183,148,246,0.15); transition:all 0.3s ease;'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:40px; height:40px; background:linear-gradient(135deg, rgba(183,148,246,0.3), rgba(183,148,246,0.15)); 
                             border-radius:10px; display:flex; align-items:center; justify-content:center; 
                             margin-right:16px; flex-shrink:0; border:1px solid rgba(183,148,246,0.2);'>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="#b794f6" stroke-width="2.5" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#e0b3ff; font-size:16px; margin:0 0 8px 0; font-weight:700;'>Kecepatan Tinggi</h4>
                            <p style='color:rgba(255,255,255,0.65); font-size:13px; margin:0; line-height:1.6;'>
                                Proses deteksi dan klasifikasi hanya membutuhkan beberapa detik dengan inference time yang optimal
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:rgba(103,198,244,0.08); padding:20px; border-radius:12px; margin-bottom:16px; 
                     border:1px solid rgba(103,198,244,0.15);'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:40px; height:40px; background:linear-gradient(135deg, rgba(103,198,244,0.3), rgba(103,198,244,0.15)); 
                             border-radius:10px; display:flex; align-items:center; justify-content:center; 
                             margin-right:16px; flex-shrink:0; border:1px solid rgba(103,198,244,0.2);'>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke="#67c6f4" stroke-width="2.5" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M22 4L12 14.01l-3-3" stroke="#67c6f4" stroke-width="2.5" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#9ed7f5; font-size:16px; margin:0 0 8px 0; font-weight:700;'>Akurasi Presisi</h4>
                            <p style='color:rgba(255,255,255,0.65); font-size:13px; margin:0; line-height:1.6;'>
                                Dual-model architecture dengan YOLO untuk deteksi dan CNN untuk klasifikasi memberikan hasil yang sangat akurat
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:rgba(159,122,234,0.08); padding:20px; border-radius:12px; margin-bottom:16px; 
                     border:1px solid rgba(159,122,234,0.15);'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:40px; height:40px; background:linear-gradient(135deg, rgba(159,122,234,0.3), rgba(159,122,234,0.15)); 
                             border-radius:10px; display:flex; align-items:center; justify-content:center; 
                             margin-right:16px; flex-shrink:0; border:1px solid rgba(159,122,234,0.2);'>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" 
                                      stroke="#9f7aea" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <polyline points="3.27 6.96 12 12.01 20.73 6.96" stroke="#9f7aea" stroke-width="2" 
                                          stroke-linecap="round" stroke-linejoin="round"/>
                                <line x1="12" y1="22.08" x2="12" y2="12" stroke="#9f7aea" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#c4a7f4; font-size:16px; margin:0 0 8px 0; font-weight:700;'>Multi-Object Detection</h4>
                            <p style='color:rgba(255,255,255,0.65); font-size:13px; margin:0; line-height:1.6;'>
                                Mampu mendeteksi dan mengklasifikasi multiple kendaraan dalam satu gambar secara bersamaan dengan presisi tinggi
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:rgba(72,187,120,0.08); padding:20px; border-radius:12px; margin-bottom:16px;
                     border:1px solid rgba(72,187,120,0.15);'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:40px; height:40px; background:linear-gradient(135deg, rgba(72,187,120,0.3), rgba(72,187,120,0.15)); 
                             border-radius:10px; display:flex; align-items:center; justify-content:center; 
                             margin-right:16px; flex-shrink:0; border:1px solid rgba(72,187,120,0.2);'>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" stroke="#48bb78" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M7 11V7a5 5 0 0 1 10 0v4" stroke="#48bb78" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#7ed6a3; font-size:16px; margin:0 0 8px 0; font-weight:700;'>Privasi Terjamin</h4>
                            <p style='color:rgba(255,255,255,0.65); font-size:13px; margin:0; line-height:1.6;'>
                                Gambar diproses secara lokal dan tidak disimpan di server untuk menjaga keamanan dan privasi data Anda
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background:rgba(236,201,75,0.08); padding:20px; border-radius:12px;
                     border:1px solid rgba(236,201,75,0.15);'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:40px; height:40px; background:linear-gradient(135deg, rgba(236,201,75,0.3), rgba(236,201,75,0.15)); 
                             border-radius:10px; display:flex; align-items:center; justify-content:center; 
                             margin-right:16px; flex-shrink:0; border:1px solid rgba(236,201,75,0.2);'>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" stroke="#ecc94b" stroke-width="2" 
                                          stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#f6e05e; font-size:16px; margin:0 0 8px 0; font-weight:700;'>Analisis Real-time</h4>
                            <p style='color:rgba(255,255,255,0.65); font-size:13px; margin:0; line-height:1.6;'>
                                Dapatkan statistik lengkap, confidence score, dan visualisasi data secara instant untuk keputusan cepat
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("← Kembali", use_container_width=True):
                go_prev()
        with col3:
            if st.button("Lanjut →", use_container_width=True):
                go_next()
                
#halaman 2 ----------------------------------------
elif st.session_state.page == 2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center'>Pengaturan Pengguna</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.6);margin-bottom:32px'>Personalisasi pengalaman Anda</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<h4>Nama Pengguna</h4>", unsafe_allow_html=True)
        name = st.text_input("", value=st.session_state.user_name, placeholder="Masukkan nama Anda...", label_visibility="collapsed")
        st.session_state.user_name = name
        st.markdown("<p style='color:rgba(255,255,255,0.5);font-size:13px;margin-top:8px'>Minimal 2 karakter</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Informasi Developer</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.04);padding:24px;border-radius:12px;border:1px solid rgba(255,255,255,0.1)'>
            <div style='display:flex;align-items:center;gap:16px;margin-bottom:16px'>
                <div style='width:60px;height:60px;background:linear-gradient(135deg,#7c3aed,#a78bfa);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:700;color:white'>JP</div>
                <div>
                    <p style='margin:0;color:#e0b3ff;font-weight:700;font-size:18px'>Jilan Putri Malisa</p>
                    <p style='margin:4px 0 0;color:rgba(255,255,255,0.6);font-size:14px'>AI Developer & Data Scientist</p>
                </div>
            </div>
            <div style='border-top:1px solid rgba(255,255,255,0.1);padding-top:16px'>
                <p style='color:rgba(255,255,255,0.6);font-size:14px;margin-bottom:12px'>
                    📧 jilanptr06@gmail.com
                </p>
                <div style='display:flex;gap:10px'>
                    <a href='https://github.com/jilan20' target='_blank' 
                       style='flex:1;background:rgba(167,139,250,0.1);padding:10px;border-radius:8px;
                       text-align:center;color:#a78bfa;text-decoration:none;font-weight:600;font-size:14px;
                       border:1px solid rgba(167,139,250,0.2)'>
                        GitHub
                    </a>
                    <a href='https://linkedin.com' target='_blank' 
                       style='flex:1;background:rgba(167,139,250,0.1);padding:10px;border-radius:8px;
                       text-align:center;color:#a78bfa;text-decoration:none;font-weight:600;font-size:14px;
                       border:1px solid rgba(167,139,250,0.2)'>
                        LinkedIn
                    </a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("← Kembali", use_container_width=True):
            go_prev()
    with col3:
        if st.button("Lanjutkan →", use_container_width=True):
            go_next()

#halaman 3 ------------------------------------
# Page 3 - Detection & Classification
elif st.session_state.page == 3:
    with content_col:
        if st.session_state.user_name:
            st.markdown(f"""
                <div class='welcome-banner'>
                    <h4>Selamat datang, {st.session_state.user_name}!</h4>
                    <p class='muted' style='margin:8px 0 0 0'>Unggah gambar kendaraan untuk memulai deteksi dan klasifikasi menggunakan AI</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Deteksi & Klasifikasi Kendaraan</h3>", unsafe_allow_html=True)
        st.markdown("<p class='muted'>Unggah gambar kendaraan dalam format JPG atau PNG untuk analisis otomatis dengan dual-model AI</p>", unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Pilih Gambar", type=["jpg","jpeg","png"], label_visibility="collapsed")
        
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.session_state["uploaded_image_pil"] = image
                st.session_state["uploaded_image_bytes"] = uploaded.getvalue()
            except Exception as e:
                st.error(f"Gagal membuka gambar: {str(e)}")
        
        if "uploaded_image_pil" in st.session_state:
            col_img, col_preview = st.columns([2, 1], gap="large")
            
            with col_img:
                st.image(st.session_state["uploaded_image_pil"], caption="Gambar Input", use_container_width=True)
            
            with col_preview:
                file_type = uploaded.type if uploaded else "unknown"
                file_size = f"{uploaded.size / 1024:.1f} KB" if uploaded else "0 KB"
                img_width, img_height = st.session_state["uploaded_image_pil"].size
                st.markdown(f"""
                    <div class='settings-box'>
                        <h4 style='margin:0 0 12px 0;color:#e0b3ff'>Info Gambar</h4>
                        <p class='muted' style='margin:4px 0'><strong>Format:</strong> {file_type}</p>
                        <p class='muted' style='margin:4px 0'><strong>Ukuran:</strong> {file_size}</p>
                        <p class='muted' style='margin:4px 0'><strong>Dimensi:</strong> {img_width} x {img_height}px</p>
                    </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                if yolo_model is not None and classifier is not None:
                    if st.button("Mulai Deteksi & Klasifikasi", use_container_width=True):
                        start_time = time.time()
                        with st.spinner("AI sedang menganalisis gambar..."):
                            try:
                                # YOLO Detection
                                results = yolo_model.predict(
                                    np.array(st.session_state["uploaded_image_pil"]), 
                                    imgsz=960, 
                                    conf=0.45, 
                                    max_det=50
                                )
                                r0 = results[0]
                                
                                dets = []
                                classifications = []
                                
                                if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
                                    boxes = r0.boxes.xyxy.cpu().numpy()
                                    scores = r0.boxes.conf.cpu().numpy()
                                    classes = r0.boxes.cls.cpu().numpy().astype(int)
                                    names = r0.names if hasattr(r0, "names") else {}
                                    
                                    orig_img = st.session_state["uploaded_image_pil"]
                                    
                                    # Classify each detection
                                    for idx, (b, s, c) in enumerate(zip(boxes, scores, classes)):
                                        yolo_class = names.get(int(c), str(c))
                                        
                                        # Crop the detected region
                                        x1, y1, x2, y2 = map(int, b)
                                        crop = orig_img.crop((x1, y1, x2, y2))
                                        
                                        # Classify the crop
                                        class_name, class_conf = classify_crop(crop, classifier)
                                        
                                        dets.append({
                                            "ID": idx + 1,
                                            "YOLO Class": yolo_class,
                                            "Classified As": class_name,
                                            "Detection Conf": f"{float(s):.1%}",
                                            "Classification Conf": f"{class_conf:.1%}",
                                            "Bounding Box": f"({x1}, {y1}, {x2}, {y2})",
                                            "Det_Confidence": float(s),
                                            "Class_Confidence": class_conf
                                        })
                                        
                                        classifications.append({
                                            "class": class_name,
                                            "confidence": class_conf
                                        })
                                
                                # Draw results with classifications
                                plotted = r0.plot()
                                if isinstance(plotted, np.ndarray):
                                    img = Image.fromarray(plotted)
                                else:
                                    img = plotted if isinstance(plotted, Image.Image) else Image.fromarray(np.array(plotted))
                                
                                # Resize for display
                                target_width = 900
                                max_width = 1200
                                if img.width < target_width:
                                    scale = target_width / img.width
                                    new_size = (int(img.width * scale), int(img.height * scale))
                                    img = img.resize(new_size, Image.LANCZOS)
                                if img.width > max_width:
                                    scale = max_width / img.width
                                    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
                                
                                st.session_state["result_image"] = img
                                st.session_state["dets"] = dets
                                st.session_state["classifications"] = classifications
                                
                                end_time = time.time()
                                st.session_state["process_time"] = end_time - start_time
                                st.success("Deteksi dan klasifikasi berhasil!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                else:
                    st.error("Model tidak tersedia.")
        
        if "result_image" in st.session_state and "dets" in st.session_state:
            st.markdown("</div>", unsafe_allow_html=True)

            # Display result image
            st.markdown("<div style='margin:32px 0;padding:4px;background:linear-gradient(135deg,rgba(183,148,246,0.3),rgba(139,115,209,0.2));border-radius:20px'>", unsafe_allow_html=True)
            st.image(st.session_state["result_image"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1.5,1,1.5])
            with col2:
                st.markdown("<div style='background:linear-gradient(135deg,#8b73d1,#b794f6);color:white;border:none;border-radius:12px;padding:12px 24px;font-weight:600;text-align:center'>Hasil Analisis AI</div>", unsafe_allow_html=True)
            
            # Statistics - HITUNG DARI HASIL KLASIFIKASI
            dets = st.session_state["dets"]
            
            # Hitung berdasarkan hasil KLASIFIKASI (bukan YOLO detection)
            car_count = sum(1 for d in dets if 'car' in d['Classified As'].lower())
            bike_count = sum(1 for d in dets if 'bike' in d['Classified As'].lower() or 'motor' in d['Classified As'].lower())
            total_detected = len(dets)
            
            avg_det_conf = np.mean([d['Det_Confidence'] for d in dets]) if dets else 0
            avg_class_conf = np.mean([d['Class_Confidence'] for d in dets]) if dets else 0
            process_time = st.session_state.get("process_time", 0)
            
            st.markdown("<div class='card' style='margin-top:32px'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-bottom:24px;text-align:center'>Statistik Deteksi & Klasifikasi</h3>", unsafe_allow_html=True)

            stat1, stat2, stat3, stat4, stat5, stat6 = st.columns(6, gap="medium")
            
            total_vehicles = car_count + bike_count
            car_pct = (car_count / total_vehicles * 100) if total_vehicles > 0 else 0
            bike_pct = (bike_count / total_vehicles * 100) if total_vehicles > 0 else 0
            
            with stat1:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-label'>Total Objek</div>
                        <div class='stat-number'>{total_detected}</div>
                        <div class='stat-sublabel'>Terdeteksi</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat2:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-label'>Akurasi Deteksi</div>
                        <div class='stat-number'>{avg_det_conf:.0%}</div>
                        <div class='stat-sublabel'>YOLO Model</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat3:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-label'>Akurasi Klasifikasi</div>
                        <div class='stat-number'>{avg_class_conf:.0%}</div>
                        <div class='stat-sublabel'>CNN Model</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat4:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-label'>Mobil</div>
                        <div class='stat-number'>{car_count}</div>
                        <div class='stat-sublabel'>{car_pct:.0f}% dari total</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat5:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-label'>Motor</div>
                        <div class='stat-number'>{bike_count}</div>
                        <div class='stat-sublabel'>{bike_pct:.0f}% dari total</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat6:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-label'>Waktu Proses</div>
                        <div class='stat-number'>{process_time:.2f}s</div>
                        <div class='stat-sublabel'>Total inferensi</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

            # Detailed table
            st.markdown("<div class='detail-table'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-bottom:20px'>Detail Klasifikasi Objek</h3>", unsafe_allow_html=True)
            st.markdown("<div class='table-header'><div>ID</div><div>Kelas</div><div>Conf. Deteksi</div><div>Conf. Klasifikasi</div></div>", unsafe_allow_html=True)
            
            for d in dets:
                badge_class = "badge-car" if "car" in d["Classified As"].lower() else "badge-bike"
                st.markdown(f"""
                    <div class='table-row'>
                        <div style='color:#c7bfe8;font-weight:700'>#{d['ID']}</div>
                        <div><span class='class-badge {badge_class}'>{d['Classified As'].title()}</span></div>
                        <div>
                            <div style='color:#c7bfe8;font-weight:600'>{d['Detection Conf']}</div>
                            <div class='conf-bar' style='width:100%;margin-top:4px;background:rgba(255,255,255,0.1);height:6px;border-radius:3px;overflow:hidden'>
                                <div class='conf-fill' style='width:{d["Det_Confidence"]*100}%;height:100%;background:linear-gradient(90deg,#7c3aed,#a78bfa)'></div>
                            </div>
                        </div>
                        <div>
                            <div style='color:#b8aed4;font-weight:600'>{d['Classification Conf']}</div>
                            <div class='conf-bar' style='width:100%;margin-top:4px;background:rgba(255,255,255,0.1);height:6px;border-radius:3px;overflow:hidden'>
                                <div class='conf-fill' style='width:{d["Class_Confidence"]*100}%;height:100%;background:linear-gradient(90deg,#67c6f4,#9ed7f5)'></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download CSV
            try:
                df = pd.DataFrame(dets)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Laporan CSV", data=csv, file_name="vehicle_classification_results.csv", mime="text/csv", use_container_width=True)
            except Exception:
                pass
            
            # Navigation
            col_l, col_r = st.columns([1,1])
            with col_l:
                if st.button("← Kembali", key="back_results"):
                    go_prev()
            with col_r:
                if st.button("Mulai Baru", key="reset_results"):
                    keys_to_clear = ["uploaded_image_pil", "uploaded_image_bytes", "result_image", "dets", "classifications", "process_time"]
                    for k in keys_to_clear:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
        
        # Feedback section
        st.markdown("<div class='card' style='margin-top:32px'>", unsafe_allow_html=True)
        st.markdown("<h3>Feedback & Rating</h3>", unsafe_allow_html=True)
        st.markdown("<p class='muted'>Bantu kami meningkatkan sistem dengan memberikan feedback Anda</p>", unsafe_allow_html=True)
        
        feedback = st.text_area("Tulis feedback Anda", placeholder="Bagikan pengalaman, saran, atau kritik Anda tentang aplikasi ini...", label_visibility="collapsed", height=100)
        
        col_rate, col_submit = st.columns([2,1])
        with col_rate:
            rating = st.select_slider("Rating Aplikasi", options=[1,2,3,4,5], value=5)
        with col_submit:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("Kirim Feedback", use_container_width=True):
                if feedback:
                    st.success("Terima kasih atas feedback Anda!")
                else:
                    st.warning("Mohon tulis feedback terlebih dahulu")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("← Kembali"):
                go_prev()
        with col3:
            if st.button("Mulai Baru"):
                for key in ["uploaded_image_pil", "uploaded_image_bytes", "result_image", "dets", "classifications", "process_time"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
