import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import fitz  # PyMuPDF
import re
import shap
import spacy
import json
import os
from supabase import create_client, Client

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Legal AI | Case Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        max-width: 100%;
        padding-left: 5%;
        padding-right: 5%;
    }
    
    /* Navbar Clipping Fix */
    [data-testid="stHorizontalBlock"]:first-of-type {
        margin-top: 15px;
    }

    /* Dashboard Navbar Logout Button styling */
    [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:nth-of-type(4) [data-testid="stButton"] button {
        background-color: #ef4444 !important; 
        color: white !important; 
        border: none !important;
    }
    [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:nth-of-type(4) [data-testid="stButton"] button:hover {
        background-color: #dc2626 !important; 
        transform: scale(1.05) !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* Hide Sidebar & Toggle */
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    
    /* Global Animations & Transitions for all Buttons */
    .stButton > button {
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2), 0 4px 6px -4px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Table Rows Hover Scale */
    [data-testid="stDataFrame"] {
        transition: transform 0.3s ease;
    }
    [data-testid="stDataFrame"]:hover {
        transform: scale(1.01);
    }
    

    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #2563eb;
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    
    /* Hero Section */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        color: #0f172a;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: #64748b;
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .divider {
        height: 1px;
        background: linear-gradient(to right, #e2e8f0, transparent);
        margin: 1.5rem 0 2.5rem 0;
    }
    
    /* File Uploader Container */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: 1px dashed #94a3b8 !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #f1f5f9 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background-color: #ffffff !important; 
        color: #0f172a !important; 
        border: 1px solid #cbd5e1 !important; 
        font-weight: normal !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    /* Metric Cards */
    .metric-card-container {
        display: flex;
        flex-direction: column;
        background: #ffffff;
        border-radius: 12px;
        padding: 1.75rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        border: 1px solid #f1f5f9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        margin-bottom: 1rem;
    }
    .metric-card-container:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
        border-color: #cbd5e1;
    }
    .metric-card-title {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    .metric-card-value {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .metric-card-sub {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #64748b;
        font-size: 1.05rem;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom-color: #2563eb !important;
        border-bottom-width: 3px !important;
    }
    
    /* Dataframes/Tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATABASE & AUTH INITIALIZATION ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase: Client = init_connection()
except Exception as e:
    st.error("Failed to connect to Supabase. Please check your .streamlit/secrets.toml file.")
    st.stop()

# --- 3. SESSION STATE INITIALIZATION ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'cleaned_text' not in st.session_state:
    st.session_state.cleaned_text = ""
if 'critical_chunk' not in st.session_state:
    st.session_state.critical_chunk = ""
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'structured_data' not in st.session_state:
    st.session_state.structured_data = {}
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Analysis"

# --- 4. INFERENCE PIPELINE (LOCAL MODELS) ---
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT", use_fast=False)
    # Ensure these paths correctly point to your local models
    model_b = AutoModelForSequenceClassification.from_pretrained("./models/Module_B/Final")
    model_c = AutoModelForSequenceClassification.from_pretrained("./models/Module_C/Final")
    return tokenizer, model_b, tokenizer, model_c

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# --- 5. DATA PRE-PROCESSING & RESEARCH UPGRADES ---
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def clean_legal_text(text):
    pattern = r"(?i)^.*?(?:\bHEADNOTE\b[\s:\"-]*|\n(?<!DATE OF )\bJUDGMENT\b[\s:\"-]*|\n\s*ORDER\s*[\s:\"-]*)(\n|$)"
    cleaned = re.sub(pattern, '', text, count=1, flags=re.DOTALL)
    if len(cleaned) == len(text):
        fallback = r"(?im)^(Equivalent citations|Bench|PETITIONER|RESPONDENT|DATE OF JUDGMENT|CITATION|CITATOR INFO|ACT)[\s\S]*?(?=\n\n|\bHEADNOTE\b)"
        cleaned = re.sub(fallback, "", cleaned)
    return cleaned.strip()

# 🚀 RESEARCH UPGRADE 1: DYNAMIC JSON MAPPING WITH REGEX WORD BOUNDARIES
@st.cache_data
def load_bns_mapping():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_path, "bns_mapping.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load BNS mapping JSON file: {e}. Ensure 'bns_mapping.json' is in the same folder.")
        return {}

ipc_to_bns_map = load_bns_mapping()

def translate_laws_to_bns(text):
    if not ipc_to_bns_map:
        return text 
        
    for ipc, bns in ipc_to_bns_map.items():
        # Added \b (Word Boundary) to prevent partial matches like "Section 30" hitting "Section 302"
        pattern = re.compile(r'\b' + re.escape(ipc) + r'\b', re.IGNORECASE)
        text = pattern.sub(f" [{bns}] ", text)
    return text

# 🚀 RESEARCH UPGRADE 2: STRUCTURE-AWARE CHUNKING
def structure_aware_chunking(text):
    sections = {
        "FACTS": r"(?i)(?:brief facts|factual matrix|facts of the case|factual aspect)[\s\S]*?(?=\n(?:arguments|submissions|issues|judgment|reasons|analysis))",
        "ARGUMENTS": r"(?i)(?:arguments|submissions|learned counsel for|rival submissions)[\s\S]*?(?=\n(?:issues|court observations|judgment|reasons|analysis))",
        "OBSERVATIONS": r"(?i)(?:court observations|reasons|analysis)[\s\S]*?(?=\n(?:final judgment|order|conclusion|held))",
        "JUDGMENT": r"(?i)(?:final judgment|order|conclusion|held)[\s\S]*"
    }
    
    extracted = {}
    for sec, pat in sections.items():
        match = re.search(pat, text)
        extracted[sec] = match.group(0).strip() if match else ""
    
    critical_text = extracted.get("FACTS", "") + " " + extracted.get("ARGUMENTS", "")
    
    if len(critical_text.strip()) < 100:
        critical_text = " ".join(text.split()[:400])
        
    return extracted, critical_text

def predict(text, tokenizer, model, label_map):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class_id].item() * 100
    return label_map[predicted_class_id], confidence

# 🚀 RESEARCH UPGRADE 3: NER-FILTERED EXPLAINABLE AI
def generate_shap_visuals(text, model, tokenizer):
    short_text = " ".join(text.split()[:200])
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
    explainer = shap.Explainer(pipe)
    shap_values = explainer([short_text])
    
    if nlp:
        doc = nlp(short_text)
        legal_entities = [ent.text.lower() for ent in doc.ents]
        legal_entities.extend([
            "section", "bns", "ipc", "bail", "appeal", "dismissed", 
            "allowed", "court", "judge", "guilty", "convicted", "acquitted",
            "fir", "police", "evidence", "murder", "conspiracy", "cheating"
        ])
        
        for i, token in enumerate(shap_values.data[0]):
            clean_token = token.strip().lower()
            is_entity = any(clean_token in entity for entity in legal_entities)
            if not is_entity:
                shap_values.values[0][i] = 0.0 
                
    return shap.plots.text(shap_values, display=False)

# --- 6. AUTHENTICATION UI (FULL PAGE) ---
def render_login_page():
    # Inject Full-Screen Dark Mode & Glass-Morphism CSS only for Login
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a !important;
        background-image: radial-gradient(circle at 50% 10%, #1e293b 0%, #0f172a 100%) !important;
    }
    
    .massive-hero {
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        font-size: 4.5rem;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #eab308 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.25rem;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }
    
    /* Login inputs specific override */
    [data-testid="stTextInput"] label {
        color: #e2e8f0 !important;
    }
    [data-testid="stTextInput"] input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #eab308 !important;
        box-shadow: 0 0 0 1px #eab308 !important;
    }
    
    /* Glass Morphism Card on the middle column */
    [data-testid="column"]:nth-of-type(2) {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* Target specifically the Log In button */
    [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(1) [data-testid="stButton"] button {
        background-color: #facc15 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(250, 204, 21, 0.4) !important;
        border: none !important;
    }
    [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(1) [data-testid="stButton"] button:hover {
        background-color: #eab308 !important;
        box-shadow: 0 6px 20px rgba(250, 204, 21, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 5vh;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='massive-hero'>⚖️ Legal AI Engine 4.0</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>The Next-Generation Secure Lawyer Portal</div>", unsafe_allow_html=True)
    
    # Centered layout using columns
    _, col2, _ = st.columns([1, 1.2, 1])
    
    with col2:
        email = st.text_input("Email Address", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("Log In", use_container_width=True):
                if email and password:
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user_email = res.user.email
                        st.session_state.current_page = "Analysis"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login Failed: {e}")
                else:
                    st.warning("Please enter email and password.")
                    
        with btn_col2:
            if st.button("Sign Up", use_container_width=True):
                if email and password:
                    try:
                        res = supabase.auth.sign_up({"email": email, "password": password})
                        st.success("Account created! You can now log in.")
                    except Exception as e:
                        st.error(f"Signup Failed: {e}")
                else:
                    st.warning("Please enter email and password.")

# --- 7. MAIN APPLICATION UI ---
if st.session_state.user_email is None:
    render_login_page()
else:
    # --- TOP NAVBAR ---
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2.5, 0.5, 0.5, 0.5])
    with nav_col1:
        st.markdown("<div style='font-size: 2.25rem; font-weight: 900; color: #0f172a; display: flex; align-items: center; padding-top: 0.2rem; letter-spacing: -0.04em;'><span style='font-size: 2.5rem; margin-right: 0.5rem;'>⚖️</span> Legal AI Engine <span style='color: #2563eb; margin-left: 0.5rem;'>4.0</span></div>", unsafe_allow_html=True)
    with nav_col2:
        if st.button("Analysis", use_container_width=True):
            st.session_state.current_page = "Analysis"
            st.rerun()
    with nav_col3:
        if st.button("History", use_container_width=True):
            st.session_state.current_page = "History"
            st.rerun()
    with nav_col4:
        if st.button("Logout", use_container_width=True, type="primary"):
            supabase.auth.sign_out()
            st.session_state.user_email = None
            st.session_state.analyzed = False
            st.session_state.current_page = "Analysis"
            st.rerun()
            
    st.markdown("<div style='height: 1px; background: #e2e8f0; margin-top: 1rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    try:
        if st.session_state.current_page == "Analysis":
            with st.spinner("⚖️ AI Engine verifying legal precedents & loading models..."):
                tokenizer_b, model_b, tokenizer_c, model_c = load_models()
    except Exception as e:
        st.error(f"Model Loading Error. Details: {e}")
        st.stop()

    if st.session_state.current_page == "Analysis":
        st.markdown("<div class='hero-subtitle' style='margin-bottom: 2.5rem;'>Advanced AI utilizing Structure-Aware Chunking, IPC-BNS mapping, and Secure Case Storage.</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])

        if uploaded_file and uploaded_file.name != st.session_state.get('last_uploaded_file'):
            st.session_state.analyzed = False
            st.session_state.last_uploaded_file = uploaded_file.name

        if uploaded_file is not None:
            if st.button("Analyze Document", use_container_width=True):
                with st.spinner("⚖️ AI Engine orchestrating Extraction & Research Pipeline..."):
                    if uploaded_file.type == "application/pdf":
                        raw_text = extract_text_from_pdf(uploaded_file)
                    else:
                        raw_text = uploaded_file.getvalue().decode("utf-8")
                        
                    cleaned_text = clean_legal_text(raw_text)
                    mapped_text = translate_laws_to_bns(cleaned_text)
                    structured_data, critical_text = structure_aware_chunking(mapped_text)
                    
                    if len(critical_text) < 50:
                        st.warning("Insufficient legal facts extracted. Document may be malformed.")
                    else:
                        category_map = {0: "Civil Law", 1: "Criminal Law", 2: "Constitutional Law"}
                        category, cat_conf = predict(critical_text, tokenizer_b, model_b, category_map)
                        
                        outcome_map = {0: "Dismissed / Rejected", 1: "Allowed / Accepted"}
                        outcome, out_conf = predict(critical_text, tokenizer_c, model_c, outcome_map)
                        
                        # Save to Database
                        try:
                            supabase.table("case_predictions").insert({
                                "user_email": st.session_state.user_email,
                                "filename": uploaded_file.name,
                                "predicted_jurisdiction": category,
                                "jurisdiction_confidence": cat_conf,
                                "predicted_outcome": outcome,
                                "outcome_confidence": out_conf
                            }).execute()
                        except Exception as e:
                            st.error(f"Failed to save case to database: {e}")
                        
                        st.session_state.cleaned_text = mapped_text
                        st.session_state.critical_chunk = critical_text
                        st.session_state.structured_data = structured_data
                        st.session_state.predictions = {
                            'category': category, 'cat_conf': cat_conf,
                            'outcome': outcome, 'out_conf': out_conf
                        }
                        st.session_state.analyzed = True

        if st.session_state.analyzed:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-card-title">Predicted Jurisdiction</div>
                    <div class="metric-card-value" style="color: #0f172a;">{st.session_state.predictions['category']}</div>
                    <div class="metric-card-sub">Confidence: {st.session_state.predictions['cat_conf']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                is_allowed = "Accepted" in st.session_state.predictions['outcome'] or "Allowed" in st.session_state.predictions['outcome']
                color = "#16a34a" if is_allowed else "#dc2626"
                st.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-card-title">Predicted Appellate Outcome</div>
                    <div class="metric-card-value" style="color: {color};">{st.session_state.predictions['outcome']}</div>
                    <div class="metric-card-sub">Confidence: {st.session_state.predictions['out_conf']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🔍 Document Structure Analysis (AI Extracted)")
            tab1, tab2, tab3 = st.tabs(["Extracted Facts", "Extracted Arguments", "Full Mapped Payload"])
            
            with tab1:
                if st.session_state.structured_data.get("FACTS"):
                    st.write(st.session_state.structured_data["FACTS"][:1500] + "...")
                else:
                    st.info("Specific 'Facts' heading not detected. Used fallback chunking.")
                    
            with tab2:
                if st.session_state.structured_data.get("ARGUMENTS"):
                    st.write(st.session_state.structured_data["ARGUMENTS"][:1500] + "...")
                else:
                    st.info("Specific 'Arguments' heading not detected.")

            with tab3:
                st.markdown("*Notice how IPC references have been automatically converted to BNS equivalents in brackets.*")
                st.write(st.session_state.cleaned_text[:1500] + "...")

            st.markdown("---")
            
            if st.button("Generate AI Decision Drivers (Takes ~3 minutes)"):
                with st.spinner("⚖️ AI Engine extracting transparent decision drivers (SHAP)..."):
                    shap_html_b = generate_shap_visuals(st.session_state.critical_chunk, model_b, tokenizer_b)
                    shap_html_c = generate_shap_visuals(st.session_state.critical_chunk, model_c, tokenizer_c)
                    
                    st.markdown("### Decision Drivers")
                    st.markdown("*Red words pushed the AI towards the final prediction. Blue words pushed it away.*")
                    
                    st.markdown("**Jurisdiction Drivers:**")
                    components.html(shap_html_b, height=250, scrolling=True)
                    
                    st.markdown("**Outcome Drivers:**")
                    components.html(shap_html_c, height=250, scrolling=True)

    elif st.session_state.current_page == "History":
        st.markdown("<h3 style='color:#0f172a; margin-bottom:0.1rem;'>🗄️ Secure Case History</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; margin-bottom:1.5rem;'>Review your past AI predictions safely stored in your encrypted cloud vault.</p>", unsafe_allow_html=True)
        
        if st.button("Refresh History"):
            try:
                response = supabase.table("case_predictions").select("*").eq("user_email", st.session_state.user_email).order('created_at', desc=True).execute()
                
                if response.data:
                    st.dataframe(
                        response.data, 
                        column_order=("created_at", "filename", "predicted_jurisdiction", "predicted_outcome", "outcome_confidence"),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("No cases analyzed yet. Go to the Analysis tab to upload your first document!")
            except Exception as e:
                st.error(f"Could not load history: {e}")