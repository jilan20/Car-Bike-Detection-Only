import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import tensorflow as tf
import os
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
    
    /* FORCE HIDE ALL STREAMLIT MENU - HIGHEST PRIORITY */
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
      grid-template-columns: 1fr 1fr 2fr 1fr;
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
      grid-template-columns: 1fr 1fr 2fr 1fr;
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

    /* Result Summary */
    .result-summary {
      background: linear-gradient(135deg, rgba(167, 139, 250, 0.1), rgba(124, 58, 237, 0.05));
      border-radius: 16px;
      padding: 28px;
      margin: 24px 0;
      border: 1px solid rgba(255, 255, 255, 0.1);
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
    
    /* Hover states for links */
    a:hover {
      opacity: 0.8;
      transform: translateY(-2px);
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
     /* Hide Streamlit Footer & Header */
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    
    header {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Hide that annoying top bar */
    .stApp > header {
        background-color: transparent !important;
        display: none !important;
    }
    
    /* Remove padding at top */
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Hide deploy button and hamburger menu */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

#Model ---------------------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  #Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  #Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

#Header ---------------------------------------------------------------------------------------------------------------------
st.markdown("<div class='hero-title'>Car & Bike Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Platform deteksi serta klasifikasi kendaraan berbasis AI menggunakan teknologi Computer Vision dan Deep Learning</div>", unsafe_allow_html=True)

content_col = st.container()

#Halaman 0-----------------------------------------------------------------------------------------------------------
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
                    <h3>Analisis Detail</h3>
                    <p class='muted'>Tampilkan statistik lengkap: jumlah kendaraan, confidence score, dan laporan terstruktur</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='text-align:center;margin-top:16px'><p class='muted' style='font-size:14px'> Gratis · Tanpa Registrasi · Hasil Instan · Privasi Terjamin</p></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Mulai Deteksi", use_container_width=True):
                go_next()

#Halaman 1---------------------------------------------------------------------------------------------------------------------------------------
elif st.session_state.page == 1:
    with content_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")
        
        with left:
            st.markdown("<h3 style='color:#fff; font-size:22px; margin-bottom:16px; font-weight:600;'>Bagaimana Sistem Bekerja?</h3>", unsafe_allow_html=True)
            
            st.markdown("<div style='background:rgba(255,255,255,0.03); padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
            
            #Input Processing
            st.markdown("""
                <div style='margin-bottom:18px; padding-left:12px; border-left:3px solid #b794f6;'>
                    <h4 style='color:#b794f6; font-size:16px; margin-bottom:6px; font-weight:600;'>Input Processing</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Gambar diunggah dan dinormalisasi untuk memastikan format dan ukuran sesuai standar model
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            #AI Detection
            st.markdown("""
                <div style='margin-bottom:18px; padding-left:12px; border-left:3px solid #67c6f4;'>
                    <h4 style='color:#67c6f4; font-size:16px; margin-bottom:6px; font-weight:600;'>AI Detection</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Model YOLOv8n menganalisis gambar dan memprediksi lokasi setiap kendaraan dengan akurasi tinggi
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            #Classification
            st.markdown("""
                <div style='margin-bottom:18px; padding-left:12px; border-left:3px solid #9f7aea;'>
                    <h4 style='color:#9f7aea; font-size:16px; margin-bottom:6px; font-weight:600;'>Classification</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Sistem mengidentifikasi dan mengklasifikasikan jenis kendaraan (mobil atau motor)
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            #Visualization
            st.markdown("""
                <div style='padding-left:12px; border-left:3px solid #48bb78;'>
                    <h4 style='color:#48bb78; font-size:16px; margin-bottom:6px; font-weight:600;'>Visualization</h4>
                    <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                        Hasil ditampilkan dengan bounding box berwarna dan statistik lengkap untuk analisis
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with right:
            st.markdown("<h3 style='color:#fff; font-size:20px; margin-bottom:20px; font-weight:600;'>Keunggulan Sistem</h3>", unsafe_allow_html=True)
            #Kecepatan Tinggi
            st.markdown("""
                <div style='background:rgba(183,148,246,0.1); padding:20px; border-radius:10px; margin-bottom:16px; 
                     border-left:3px solid #b794f6;'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:36px; height:36px; background:rgba(183,148,246,0.2); border-radius:8px; 
                             display:flex; align-items:center; justify-content:center; margin-right:14px; flex-shrink:0;'>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="#b794f6" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#fff; font-size:16px; margin:0 0 6px 0; font-weight:600;'>Kecepatan Tinggi</h4>
                            <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                                Proses deteksi hanya membutuhkan beberapa detik dengan inference time yang optimal
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            #Akurasi Presisi
            st.markdown("""
                <div style='background:rgba(103,198,244,0.1); padding:20px; border-radius:10px; margin-bottom:16px; 
                     border-left:3px solid #67c6f4;'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:36px; height:36px; background:rgba(103,198,244,0.2); border-radius:8px; 
                             display:flex; align-items:center; justify-content:center; margin-right:14px; flex-shrink:0;'>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke="#67c6f4" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M22 4L12 14.01l-3-3" stroke="#67c6f4" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#fff; font-size:16px; margin:0 0 6px 0; font-weight:600;'>Akurasi Presisi</h4>
                            <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                                Model terlatih dengan ribuan dataset untuk memberikan hasil deteksi yang akurat
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            #Privasi Terjamin
            st.markdown("""
                <div style='background:rgba(72,187,120,0.1); padding:20px; border-radius:10px; border-left:3px solid #48bb78;'>
                    <div style='display:flex; align-items:start;'>
                        <div style='width:36px; height:36px; background:rgba(72,187,120,0.2); border-radius:8px; 
                             display:flex; align-items:center; justify-content:center; margin-right:14px; flex-shrink:0;'>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" stroke="#48bb78" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M7 11V7a5 5 0 0 1 10 0v4" stroke="#48bb78" stroke-width="2" 
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <h4 style='color:#fff; font-size:16px; margin:0 0 6px 0; font-weight:600;'>Privasi Terjamin</h4>
                            <p style='color:rgba(255,255,255,0.7); font-size:14px; margin:0; line-height:1.5;'>
                                Gambar diproses secara lokal dan tidak disimpan di server untuk menjaga keamanan data
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
                

#Halaman 2 ---------------------------------------------------------------------------------------------------------------------------------
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
        <div class='dev-box'>
            <div style='display:flex;align-items:center;gap:16px;margin-bottom:16px'>
                <div class='avatar'>JP</div>
                <div>
                    <p style='margin:0;color:#e0b3ff;font-weight:700;font-size:18px'>Jilan Putri Malisa</p>
                    <p style='margin:4px 0 0;color:rgba(255,255,255,0.6);font-size:14px'>AI Developer & Data Scientist (Aamiin ya Allah, Aamiinin bareng bareng ya teman teman)</p>
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

#Halaman 3 ----------------------------------------------------------------------------------------------------------------
elif st.session_state.page == 3:
    with content_col:
        #Welcome banner
        if st.session_state.user_name:
            st.markdown(f"""
                <div class='welcome-banner'>
                    <h4>👋 Selamat datang, {st.session_state.user_name}!</h4>
                    <p class='muted' style='margin:8px 0 0 0'>Unggah gambar kendaraan untuk memulai deteksi dan klasifikasi kendaraan menggunakan AI</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Deteksi Kendaraan</h3>", unsafe_allow_html=True)
        st.markdown("<p class='muted'>Unggah gambar kendaraan dalam format JPG atau PNG untuk analisis otomatis</p>", unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Pilih Gambar", type=["jpg","jpeg","png"], label_visibility="collapsed")
        
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.session_state["uploaded_image_pil"] = image
                st.session_state["uploaded_image_bytes"] = uploaded.getvalue()
            except Exception as e:
                st.error(f"❌ Gagal membuka gambar: {str(e)}")
        
        if "uploaded_image_pil" in st.session_state:
            col_img, col_preview = st.columns([2, 1], gap="large")
            
            with col_img:
                st.image(st.session_state["uploaded_image_pil"], caption="📸 Gambar Input", use_container_width=True)
            
            with col_preview:
                file_type = uploaded.type if uploaded else "unknown"
                file_size = f"{uploaded.size / 1024:.1f} KB" if uploaded else "0 KB"
                st.markdown(f"""
                    <div class='settings-box'>
                        <h4 style='margin:0 0 12px 0;color:#e0b3ff'>Info Gambar</h4>
                        <p class='muted' style='margin:4px 0'><strong>Format:</strong> {file_type}</p>
                        <p class='muted' style='margin:4px 0'><strong>Ukuran:</strong> {file_size}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            #tombol detect
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                if yolo_model is not None:
                    if st.button("🔍 Mulai Deteksi", use_container_width=True):
                        start_time = time.time()
                        with st.spinner("🤖 Model sedang menganalisis gambar..."):
                            try:
                                results = yolo_model.predict(
                                    np.array(st.session_state["uploaded_image_pil"]), 
                                    imgsz=960, 
                                    conf=0.45, 
                                    max_det=50
                                )
                                r0 = results[0]
                                
                                plotted = r0.plot()  
                                if isinstance(plotted, np.ndarray):
                                    img = Image.fromarray(plotted)
                                else:
                                    img = plotted if isinstance(plotted, Image.Image) else Image.fromarray(np.array(plotted))

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

                                dets = []
                                if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
                                    boxes = r0.boxes.xyxy.cpu().numpy()
                                    scores = r0.boxes.conf.cpu().numpy()
                                    classes = r0.boxes.cls.cpu().numpy().astype(int)
                                    names = r0.names if hasattr(r0, "names") else {}
                                    
                                    for b, s, c in zip(boxes, scores, classes):
                                        class_name = names.get(int(c), str(c))
                                        dets.append({
                                            "Kelas": class_name,
                                            "Akurasi": f"{float(s):.1%}",
                                            "Bounding Box": f"({int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])})",
                                            "Confidence": float(s)
                                        })
                                
                                end_time = time.time()
                                st.session_state["dets"] = dets
                                st.session_state["process_time"] = end_time - start_time
                                st.success("✅ Deteksi berhasil!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error saat inferensi: {str(e)}")
                else:
                    st.error("⚠️ Model tidak tersedia. Pastikan file model tersedia di folder yang benar.")
        
        if "result_image" in st.session_state and "dets" in st.session_state:
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin:32px 0;padding:4px;background:linear-gradient(135deg,rgba(183,148,246,0.3),rgba(139,115,209,0.2));border-radius:20px'>", unsafe_allow_html=True)
            st.image(st.session_state["result_image"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1.5,1,1.5])
            with col2:
                st.markdown("<button style='background:linear-gradient(135deg,#8b73d1,#b794f6);color:white;border:none;border-radius:12px;padding:12px 24px;font-weight:600;width:100%;cursor:default'>Hasil Deteksi</button>", unsafe_allow_html=True)
            
            #Hasil Statistik
            dets = st.session_state["dets"]
            car_count = sum(1 for d in dets if 'car' in d['Kelas'].lower())
            bike_count = sum(1 for d in dets if 'bike' in d['Kelas'].lower() or 'motor' in d['Kelas'].lower())
            avg_conf = np.mean([d['Confidence'] for d in dets]) if dets else 0
            process_time = st.session_state.get("process_time", 0)
            
            st.markdown("<div class='card' style='margin-top:32px'>", unsafe_allow_html=True)

            stat1, stat2, stat3, stat4 = st.columns(4, gap="medium")
            
            total_vehicles = car_count + bike_count
            car_pct = (car_count / total_vehicles * 100) if total_vehicles > 0 else 0
            bike_pct = (bike_count / total_vehicles * 100) if total_vehicles > 0 else 0
            
            with stat1:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div style='font-size:24px;margin-bottom:8px'>✅</div>
                        <div class='stat-label'>Akurasi</div>
                        <div class='stat-number'>{avg_conf:.1%}</div>
                        <div class='accuracy-bar'>
                            <div class='progress-track'>
                                <div class='progress-fill' style='width:{avg_conf*100}%'>{avg_conf:.1%}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat2:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div style='font-size:24px;margin-bottom:8px'>🚗</div>
                        <div class='stat-label'>Mobil</div>
                        <div class='stat-number'>{car_count}</div>
                        <div class='stat-sublabel'>{car_pct:.0f}% dari total</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat3:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div style='font-size:24px;margin-bottom:8px'>🏍️</div>
                        <div class='stat-label'>Motor</div>
                        <div class='stat-number'>{bike_count}</div>
                        <div class='stat-sublabel'>{bike_pct:.0f}% dari total</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat4:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div style='font-size:24px;margin-bottom:8px'>⏱️</div>
                        <div class='stat-label'>Waktu</div>
                        <div class='stat-number'>{process_time:.2f}s</div>
                        <div class='stat-sublabel'>Waktu inferensi</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # close card

            st.markdown("<div class='detail-table'>", unsafe_allow_html=True)
            st.markdown("<div class='table-header'><div>Class</div><div>Confidence</div><div>Bounding Box</div><div></div></div>", unsafe_allow_html=True)
            for d in dets:
                badge_class = "badge-car" if "car" in d["Kelas"].lower() else "badge-bike"
                st.markdown(f"""
                    <div class='table-row'>
                        <div><span class='class-badge {badge_class}'>{d['Kelas']}</span></div>
                        <div style='color:#c7bfe8;font-weight:600'>{d['Akurasi']}</div>
                        <div style='color:#b8aed4'>{d['Bounding Box']}</div>
                        <div style='text-align:right;color:#9d92b8'> </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)            
            #DataFrame & CSV download
            try:
                df = pd.DataFrame(dets)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download laporan CSV", data=csv, file_name="detections.csv", mime="text/csv")
            except Exception:
                pass
            #Navigasi
            col_l, col_r = st.columns([1,1])
            with col_l:
                if st.button("⬅️ Kembali", key="back_results"):
                    go_prev()
            with col_r:
                if st.button("Mulai Baru", key="reset_results"):
                    keys_to_clear = ["uploaded_image_pil", "uploaded_image_bytes", "result_image", "dets", "process_time"]
                    for k in keys_to_clear:
                        if k in st.session_state:
                            del st.session_state[k]

                    if hasattr(st, "experimental_rerun"):
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.session_state.page = 3
                    else:
                        st.session_state.page = 3
        #Feedback
        st.markdown("<div class='card' style='margin-top:32px'>", unsafe_allow_html=True)
        st.markdown("<h3>💬 Feedback</h3>", unsafe_allow_html=True)
        feedback = st.text_area("Berikan feedback Anda tentang aplikasi ini", placeholder="Tulis feedback, saran, atau kritik Anda di sini...", label_visibility="collapsed")
        rating = st.select_slider("Rating", options=[1,2,3,4,5], value=5)
        
        if st.button("Kirim Feedback"):
            if feedback:
                st.success("✅ Terima kasih atas feedback Anda!")
            else:
                st.warning("⚠️ Mohon tulis feedback terlebih dahulu")
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("← Kembali"):
                go_prev()
        with col3:
            if st.button("Mulai Baru"):
                for key in ["img", "result_img", "dets", "time"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
