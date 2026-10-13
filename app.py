import streamlit as st

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN





# =========================================================

# PAGE CONFIGURATION

# =========================================================



st.set_page_config(

    page_title="Rare Disease Detection",

    page_icon="🧬",

    layout="wide"

)





# =========================================================

# CUSTOM CSS

# =========================================================



st.markdown(

    """

    <style>



    /* =====================================================

       SIDEBAR

       ===================================================== */



    section[data-testid="stSidebar"] {

        width: 350px !important;

        background: linear-gradient(

            180deg,

            #0f172a 0%,

            #1e293b 100%

        );

    }



    section[data-testid="stSidebar"] > div {

        width: 350px !important;

        padding-top: 1.2rem !important;

    }



    section[data-testid="stSidebar"] * {

        color: white;

    }



    section[data-testid="stSidebar"] h1 {

        font-size: 32px !important;

    }



    section[data-testid="stSidebar"] h2 {

        font-size: 25px !important;

    }



    section[data-testid="stSidebar"] h3 {

        font-size: 21px !important;

    }



    section[data-testid="stSidebar"] p {

        font-size: 16px !important;

    }





    /* =====================================================

       CLEAN NAVIGATION

       ===================================================== */



    section[data-testid="stSidebar"]

    div[data-testid="stRadio"] > label {

        font-size: 17px !important;

        font-weight: 700 !important;

        margin-bottom: 12px !important;

        letter-spacing: 0.2px;

    }



    section[data-testid="stSidebar"]

    div[role="radiogroup"] {

        gap: 5px !important;

    }



    section[data-testid="stSidebar"]

    div[role="radiogroup"] label {

        width: 100% !important;

        min-height: 46px !important;

        box-sizing: border-box !important;

        display: flex !important;

        align-items: center !important;

        background: transparent !important;

        border-radius: 8px !important;

        padding: 9px 12px !important;

        margin: 1px 0 !important;

        cursor: pointer !important;

        border: 1px solid transparent !important;

        border-left: 3px solid transparent !important;

        transition: background 0.18s ease, border-color 0.18s ease;

    }



    /* Remove radio controls completely. */

    section[data-testid="stSidebar"]

    div[role="radiogroup"] label input[type="radio"],

    section[data-testid="stSidebar"]

    div[role="radiogroup"] label > div:first-child:has(input[type="radio"]),

    section[data-testid="stSidebar"]

    div[role="radiogroup"] label > div[role="radio"],

    section[data-testid="stSidebar"]

    div[role="radiogroup"] label span[role="radio"] {

        display: none !important;

    }



    section[data-testid="stSidebar"]

    div[role="radiogroup"] label:hover {

        background: rgba(255, 255, 255, 0.055) !important;

        border-left-color: rgba(148, 163, 184, 0.45) !important;

    }



    section[data-testid="stSidebar"]

    div[role="radiogroup"] label:has(input[type="radio"]:checked) {

        background: rgba(59, 130, 246, 0.16) !important;

        border: 1px solid rgba(96, 165, 250, 0.20) !important;

        border-left: 3px solid #60a5fa !important;

        box-shadow: none !important;

    }



    section[data-testid="stSidebar"]

    div[role="radiogroup"] label p {

        font-size: 17px !important;

        font-weight: 650 !important;

        margin: 0 !important;

        line-height: 1.2 !important;

    }



    /* Cleaner sidebar separators. */

    section[data-testid="stSidebar"] hr {

        border: none !important;

        border-top: 1px solid rgba(148, 163, 184, 0.18) !important;

        margin: 18px 0 !important;

    }



    /* Compact sidebar metric cards. */

    section[data-testid="stSidebar"] div[data-testid="stMetric"] {

        background: rgba(255, 255, 255, 0.035);

        border: 1px solid rgba(148, 163, 184, 0.10);

        border-radius: 8px;

        padding: 8px 9px;

    }



    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {

        font-size: 21px !important;

    }



    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {

        font-size: 12px !important;

    }





    /* =====================================================

       MAIN HEADINGS

       ===================================================== */



    h1 {

        font-size: 38px !important;

        font-weight: 700 !important;

    }



    h2 {

        font-size: 30px !important;

    }



    h3 {

        font-size: 23px !important;

    }





    /* =====================================================

       METRICS

       ===================================================== */



    div[data-testid="stMetricValue"] {

        font-size: 30px !important;

        font-weight: 700 !important;

    }



    div[data-testid="stMetricLabel"] {

        font-size: 16px !important;

    }





    /* =====================================================

       BUTTONS

       ===================================================== */



    .stButton button {

        font-size: 17px !important;

        font-weight: 700 !important;

        padding: 11px !important;

        border-radius: 8px !important;

    }





    /* =====================================================

       INPUTS

       ===================================================== */



    .stTextInput input,

    .stNumberInput input {

        border-radius: 8px !important;

    }





    /* =====================================================

       RESULT CARDS

       ===================================================== */



    .normal-result {

        padding: 25px;

        border-radius: 12px;

        background: rgba(34, 197, 94, 0.12);

        border: 1px solid rgba(34, 197, 94, 0.35);

        margin: 10px 0 20px 0;

    }



    .unusual-result {

        padding: 25px;

        border-radius: 12px;

        background: rgba(239, 68, 68, 0.12);

        border: 1px solid rgba(239, 68, 68, 0.35);

        margin: 10px 0 20px 0;

    }



    .result-title {

        font-size: 28px;

        font-weight: 800;

        margin-bottom: 8px;

    }



    .result-text {

        font-size: 17px;

        line-height: 1.5;

    }





    /* =====================================================

       MEANING GUIDE

       ===================================================== */



    .meaning-guide {

        padding: 18px;

        border-radius: 12px;

        background: rgba(59, 130, 246, 0.08);

        border: 1px solid rgba(96, 165, 250, 0.25);

        margin: 8px 0 18px 0;

    }



    .meaning-guide-title {

        font-size: 19px;

        font-weight: 700;

        margin-bottom: 10px;

    }



    .meaning-guide-row {

        font-size: 15px;

        padding: 6px 0;

        border-bottom: 1px solid rgba(148, 163, 184, 0.15);

    }



    .meaning-guide-row:last-child {

        border-bottom: none;

    }



    /* =====================================================

       INFORMATION CARDS

       ===================================================== */



    .info-card {

        padding: 20px;

        border-radius: 12px;

        background: rgba(100, 116, 139, 0.08);

        border: 1px solid rgba(100, 116, 139, 0.18);

        min-height: 130px;

    }



    .info-card-title {

        font-size: 19px;

        font-weight: 700;

        margin-bottom: 8px;

    }



    .info-card-text {

        font-size: 15px;

        line-height: 1.5;

    }



    
    /* =====================================================
       ENHANCED APPLICATION UI
       ===================================================== */

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(59, 130, 246, 0.055), transparent 28%),
            radial-gradient(circle at 10% 90%, rgba(99, 102, 241, 0.035), transparent 24%),
            #0b0f16 !important;
    }

    .main .block-container {
        max-width: 1500px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    .main h1 {
        letter-spacing: -0.7px !important;
        line-height: 1.15 !important;
    }

    .main h2,
    .main h3 {
        letter-spacing: -0.25px !important;
    }

    .main hr {
        border: none !important;
        border-top: 1px solid rgba(148, 163, 184, 0.16) !important;
        margin: 1.25rem 0 !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.48),
            rgba(15, 23, 42, 0.34)
        ) !important;
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 12px !important;
        padding: 15px 16px !important;
        transition: transform 0.18s ease, border-color 0.18s ease,
                    box-shadow 0.18s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.30) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16) !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 44px !important;
        border-radius: 9px !important;
        border: 1px solid rgba(96, 165, 250, 0.24) !important;
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.82),
            rgba(30, 64, 175, 0.82)
        ) !important;
        color: white !important;
        box-shadow: 0 5px 16px rgba(30, 64, 175, 0.18) !important;
        transition: transform 0.16s ease, box-shadow 0.16s ease,
                    border-color 0.16s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(147, 197, 253, 0.50) !important;
        box-shadow: 0 8px 22px rgba(30, 64, 175, 0.25) !important;
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0);
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        border-radius: 9px !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
        background: rgba(15, 23, 42, 0.58) !important;
        transition: border-color 0.16s ease, box-shadow 0.16s ease !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: rgba(96, 165, 250, 0.65) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.10) !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 9px !important;
        min-height: 42px !important;
    }

    div[data-testid="stSlider"] {
        padding: 4px 2px 10px 2px !important;
    }

    div[data-testid="stSlider"] [role="slider"] {
        border: 3px solid #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.12) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border-width: 1px !important;
        padding: 12px 14px !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.10) !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 10px !important;
        background: rgba(15, 23, 42, 0.28) !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background: rgba(59, 130, 246, 0.06) !important;
    }

    .info-card {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.62),
                rgba(15, 23, 42, 0.42)
            ) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        box-shadow: 0 7px 22px rgba(0, 0, 0, 0.12) !important;
        transition: transform 0.18s ease, border-color 0.18s ease,
                    box-shadow 0.18s ease !important;
    }

    .info-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.28) !important;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18) !important;
    }

    .meaning-guide {
        background:
            linear-gradient(
                145deg,
                rgba(37, 99, 235, 0.10),
                rgba(15, 23, 42, 0.30)
            ) !important;
        box-shadow: 0 7px 20px rgba(0, 0, 0, 0.10) !important;
    }

    .normal-result,
    .unusual-result {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14) !important;
    }

    section[data-testid="stSidebar"] {
        box-shadow: 8px 0 28px rgba(0, 0, 0, 0.12) !important;
    }

    section[data-testid="stSidebar"] h1 {
        letter-spacing: -0.5px !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        opacity: 0.78;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        padding: 4px 0 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        margin: 3px 0 !important;
        min-height: 44px !important;
    }

    .app-footer {
        margin-top: 28px;
        padding: 18px 20px;
        border-top: 1px solid rgba(148, 163, 184, 0.16);
        text-align: center;
        color: rgba(226, 232, 240, 0.68);
        font-size: 13px;
        line-height: 1.6;
    }

    
/* ===== PREMIUM UI POLISH ===== */
.hero-card {
    position: relative !important;
    overflow: hidden !important;
    border: 1px solid rgba(92, 168, 255, 0.30) !important;
    background: linear-gradient(135deg, rgba(19, 32, 56, 0.98), rgba(11, 22, 40, 0.96)) !important;
    box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
.hero-card:before {
    content: "";
    position: absolute;
    width: 260px; height: 260px;
    right: -90px; top: -130px;
    border-radius: 50%;
    background: rgba(74, 144, 226, 0.13);
    filter: blur(8px);
}
.hero-card:after {
    content: "";
    position: absolute;
    width: 180px; height: 180px;
    left: 42%; bottom: -120px;
    border-radius: 50%;
    background: rgba(126, 87, 255, 0.10);
    filter: blur(12px);
}
.hero-title, .hero-subtitle, .hero-badge, .hero-status { position: relative; z-index: 2; }
.hero-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    padding: 6px 11px !important;
    border-radius: 999px !important;
    background: rgba(70, 150, 255, 0.11) !important;
    border: 1px solid rgba(91, 169, 255, 0.25) !important;
    letter-spacing: 0.4px !important;
}
.hero-status {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 7px 12px !important;
    border-radius: 999px !important;
    background: rgba(34, 197, 94, 0.08) !important;
    border: 1px solid rgba(34, 197, 94, 0.20) !important;
}
.status-dot { box-shadow: 0 0 10px rgba(34,197,94,.65) !important; }
.metric-card {
    border: 1px solid rgba(255,255,255,0.07) !important;
    background: linear-gradient(145deg, rgba(23,34,54,.95), rgba(14,24,41,.94)) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.035) !important;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
}
.metric-card:hover { transform: translateY(-3px) !important; box-shadow: 0 16px 34px rgba(0,0,0,.28) !important; border-color: rgba(96,165,250,.22) !important; }
.metric-icon { filter: drop-shadow(0 5px 12px rgba(0,0,0,.22)); }
.workflow-card {
    border: 1px solid rgba(255,255,255,.07) !important;
    background: rgba(17,28,47,.80) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.03) !important;
}
.workflow-number { box-shadow: 0 7px 18px rgba(59,130,246,.18) !important; }
.section-label { letter-spacing: 1.1px !important; text-transform: uppercase !important; }
.symptom-bar-bg { background: rgba(255,255,255,.065) !important; }
.symptom-bar { box-shadow: 0 0 10px rgba(72,148,255,.18) !important; }
/* Streamlit containers and inputs */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(255,255,255,.07) !important;
    border-radius: 16px !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
    border-radius: 11px !important;
}
button[kind="primary"] {
    border-radius: 11px !important;
    font-weight: 650 !important;
    box-shadow: 0 7px 20px rgba(59,130,246,.16) !important;
}
/* Sidebar premium finish */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,.07) !important;
    box-shadow: 10px 0 30px rgba(0,0,0,.14) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    border-radius: 10px !important;
    transition: background .16s ease, transform .16s ease !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(96,165,250,.08) !important;
    transform: translateX(2px) !important;
}

@media (max-width: 900px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .main h1 {
            font-size: 31px !important;
        }

        .main h2 {
            font-size: 25px !important;
        }
    }

    
    .hero-card {
        position: relative; overflow: hidden; padding: 28px 30px;
        margin: 4px 0 24px 0; border-radius: 16px;
        background: radial-gradient(circle at 92% 15%, rgba(96,165,250,.18), transparent 28%),
                    linear-gradient(135deg, rgba(30,41,59,.92), rgba(15,23,42,.76));
        border: 1px solid rgba(96,165,250,.20);
        box-shadow: 0 12px 34px rgba(0,0,0,.18);
    }
    .hero-badge {
        display:inline-block; padding:5px 10px; margin-bottom:10px;
        border-radius:999px; background:rgba(59,130,246,.13);
        border:1px solid rgba(96,165,250,.22); color:#bfdbfe;
        font-size:12px; font-weight:700; letter-spacing:.4px; text-transform:uppercase;
    }
    .hero-title { font-size:34px; line-height:1.15; font-weight:800; letter-spacing:-.8px; margin:0 0 8px 0; }
    .hero-subtitle { color:rgba(226,232,240,.76); font-size:15px; line-height:1.55; margin:0; max-width:850px; }
    .hero-status { margin-top:16px; font-size:13px; color:rgba(226,232,240,.72); }
    .status-dot { color:#4ade80; font-size:15px; }
    .section-label { margin:2px 0 10px 0; color:rgba(191,219,254,.82); font-size:12px; font-weight:800; letter-spacing:.8px; text-transform:uppercase; }
    .metric-card {
        min-height:118px; padding:17px 18px; border-radius:13px;
        background:linear-gradient(145deg, rgba(30,41,59,.62), rgba(15,23,42,.42));
        border:1px solid rgba(148,163,184,.15); box-shadow:0 7px 22px rgba(0,0,0,.12);
        transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;
    }
    .metric-card:hover { transform:translateY(-3px); border-color:rgba(96,165,250,.30); box-shadow:0 12px 28px rgba(0,0,0,.18); }
    .metric-icon { font-size:20px; margin-bottom:8px; }
    .metric-name { color:rgba(226,232,240,.70); font-size:13px; font-weight:650; }
    .metric-number { margin-top:3px; font-size:29px; font-weight:800; letter-spacing:-.7px; }
    .metric-card.unusual { border-color:rgba(248,113,113,.18); background:linear-gradient(145deg,rgba(127,29,29,.16),rgba(15,23,42,.44)); }
    .metric-card.rare { border-color:rgba(167,139,250,.18); }
    .metric-card.common { border-color:rgba(74,222,128,.14); }
    .workflow-card {
        min-height:145px; padding:19px; border-radius:13px;
        background:rgba(15,23,42,.38); border:1px solid rgba(148,163,184,.13);
        transition:transform .18s ease,border-color .18s ease;
    }
    .workflow-card:hover { transform:translateY(-2px); border-color:rgba(96,165,250,.25); }
    .workflow-number {
        display:inline-flex; width:28px; height:28px; align-items:center; justify-content:center;
        border-radius:8px; background:rgba(59,130,246,.16);
        border:1px solid rgba(96,165,250,.20); color:#bfdbfe;
        font-size:13px; font-weight:800; margin-bottom:12px;
    }
    .workflow-title { font-size:17px; font-weight:750; margin-bottom:7px; }
    .workflow-text { color:rgba(226,232,240,.68); font-size:13px; line-height:1.5; }
    .symptom-visual {
        padding:13px 15px; margin:6px 0; border-radius:10px;
        background:rgba(15,23,42,.34); border:1px solid rgba(148,163,184,.12);
    }
    .symptom-name { font-size:14px; font-weight:700; margin-bottom:7px; }
    .symptom-bar-bg { height:8px; border-radius:999px; background:rgba(148,163,184,.15); overflow:hidden; }
    .symptom-bar { height:100%; border-radius:999px; background:linear-gradient(90deg,#60a5fa,#818cf8); }
    .symptom-meta { display:flex; justify-content:space-between; margin-top:6px; color:rgba(226,232,240,.66); font-size:12px; }
    @media (max-width:900px) { .hero-title{font-size:28px;} .hero-card{padding:22px;} }

    </style>

    """,

    unsafe_allow_html=True

)





# =========================================================

# LOAD DATA

# =========================================================



@st.cache_data

def load_data():

    try:

        loaded_data = pd.read_csv("data.csv", sep="\t")

    except FileNotFoundError:

        st.error(

            "⚠️ Dataset could not be loaded. "

            "Please make sure data.csv exists in the project folder."

        )

        st.stop()

    except Exception as error:

        st.error(

            "⚠️ The dataset could not be read. "

            f"Please check the data.csv file. Details: {error}"

        )

        st.stop()



    required_columns = [

        "PatientID", "Age", "Fatigue", "Fever", "Weakness",

        "Seizures", "Pain", "Gender", "Symptom_Duration_Days",

        "Family_History", "Hospital_Visits_Last_Year", "Label"

    ]



    missing_columns = [

        column for column in required_columns

        if column not in loaded_data.columns

    ]



    if missing_columns:

        st.error(

            "⚠️ The dataset is missing required columns: "

            + ", ".join(missing_columns)

        )

        st.stop()



    return loaded_data





data = load_data()





# =========================================================

# PREPROCESS DATA

# =========================================================



symptoms = [

    "Fatigue",

    "Fever",

    "Weakness",

    "Seizures",

    "Pain"

]





def severity_meaning(value):

    """Convert the project UI score into its meaning-guide label."""

    value = float(value)



    if value == 0:

        return "None"

    elif value <= 2:

        return "Very Mild"

    elif value <= 4:

        return "Moderate"

    elif value <= 7:

        return "Severe"

    else:

        return "Very Severe"



for column in symptoms:

    data[column] = (

        data[column]

        .astype(str)

        .str.extract(r"(\d+)")[0]

        .astype(float)

    )





data["Gender"] = data["Gender"].map({

    "Male": 0,

    "Female": 1

})





data["Family_History"] = data["Family_History"].map({

    "No": 0,

    "Yes": 1

})





# =========================================================

# FEATURES

# =========================================================



features = [

    "Age",

    "Fatigue",

    "Fever",

    "Weakness",

    "Seizures",

    "Pain",

    "Gender",

    "Symptom_Duration_Days",

    "Family_History",

    "Hospital_Visits_Last_Year"

]





# =========================================================

# STANDARDIZE DATA

# =========================================================



scaler = StandardScaler()



X_scaled = scaler.fit_transform(

    data[features]

)





# =========================================================

# DBSCAN MODEL

# =========================================================



dbscan = DBSCAN(

    eps=1.5,

    min_samples=10

)



clusters = dbscan.fit_predict(X_scaled)





# Internal model information only.

# Cluster numbers are never shown to users.



data["Cluster"] = clusters





# =========================================================

# DATASET COUNTS

# =========================================================



total_patients = len(data)



rare_cases = len(

    data[data["Label"] == "Rare"]

)



common_cases = len(

    data[data["Label"] == "Common"]

)



unusual_cases = len(

    data[data["Cluster"] == -1]

)



normal_cases = (

    total_patients - unusual_cases

)





# =========================================================

# SIDEBAR HEADER

# =========================================================



st.sidebar.markdown(

    "# 🧬 Rare Disease"

)



st.sidebar.markdown(

    "### Case Identification System"

)



st.sidebar.markdown(

    "DBSCAN-Based Anomaly Detection"

)



st.sidebar.markdown("---")





# =========================================================

# NAVIGATION

# =========================================================



pages = [

    "🏠 Dashboard",

    "🔍 Analyze Patient",

    "➕ Add Patient",

    "👤 Patient Details",

    "👥 Patient Dataset",

    "📊 Dataset Statistics",

    "📋 Analytics"

]



page = st.sidebar.radio(

    "Navigation",

    pages

)





# =========================================================

# SIDEBAR MODEL INFORMATION

# =========================================================



st.sidebar.markdown("---")



st.sidebar.markdown(

    "### 🤖 Model Information"

)



st.sidebar.markdown(

    "**Algorithm:** DBSCAN"

)



st.sidebar.markdown(

    "**Learning Type:** Unsupervised Learning"

)



st.sidebar.markdown(

    "**Purpose:** Anomaly Detection"

)





# =========================================================

# SIDEBAR DATASET INFORMATION

# =========================================================



st.sidebar.markdown("---")



st.sidebar.markdown(

    "### 📊 Dataset Information"

)



col1, col2 = st.sidebar.columns(2)



with col1:

    st.metric(

        "Patients",

        total_patients

    )



with col2:

    st.metric(

        "Unusual Patterns",

        unusual_cases

    )



col3, col4 = st.sidebar.columns(2)



with col3:

    st.metric(

        "Rare",

        rare_cases

    )



with col4:

    st.metric(

        "Common",

        common_cases

    )





# =========================================================

# SIDEBAR PROJECT INFORMATION

# =========================================================



st.sidebar.markdown("---")



st.sidebar.markdown(

    "### 📌 Project Information"

)



st.sidebar.markdown(

    "Rare-Disease Case Identification "

    "Using DBSCAN-Based Anomaly Detection"

)



st.sidebar.markdown(

    "Machine Learning Project"

)



st.sidebar.markdown("---")



st.sidebar.caption(

    "⚠️ Educational and research purposes only."

)





# =========================================================

# 1. DASHBOARD

# =========================================================



if page == "🏠 Dashboard":

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-badge">ML Case Identification System</div>
            <div class="hero-title">🧬 Rare Disease Case Identification</div>
            <p class="hero-subtitle">
                A DBSCAN-based machine learning system for identifying unusual
                patterns in patient records using unsupervised anomaly detection.
            </p>
            <div class="hero-status">
                <span class="status-dot">●</span>
                Model active &nbsp;•&nbsp; DBSCAN &nbsp;•&nbsp; {total_patients} patient records
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-label">Dataset at a glance</div>', unsafe_allow_html=True)

    metric1, metric2, metric3, metric4 = st.columns(4)

    dashboard_metrics = [
        (metric1, "👥", "Total Patients", total_patients, ""),
        (metric2, "🧬", "Rare Cases", rare_cases, "rare"),
        (metric3, "📁", "Common Cases", common_cases, "common"),
        (metric4, "🔎", "Unusual Patterns", unusual_cases, "unusual")
    ]

    for column, icon, name, value, card_class in dashboard_metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card {card_class}">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-name">{name}</div>
                    <div class="metric-number">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🤖 Machine Learning")

    ml1, ml2, ml3 = st.columns(3)

    with ml1:
        st.markdown(
            """<div class="info-card">
                <div class="info-card-title">🔬 DBSCAN</div>
                <div class="info-card-text">A density-based unsupervised learning algorithm used to identify patterns and noise in patient data.</div>
            </div>""",
            unsafe_allow_html=True
        )

    with ml2:
        st.markdown(
            """<div class="info-card">
                <div class="info-card-title">🧠 Unsupervised Learning</div>
                <div class="info-card-text">Rare/Common labels are not used during DBSCAN training. The model learns structure from the input features.</div>
            </div>""",
            unsafe_allow_html=True
        )

    with ml3:
        st.markdown(
            """<div class="info-card">
                <div class="info-card-title">🔎 Pattern Detection</div>
                <div class="info-card-text">Records that differ from the main data distribution are identified as unusual patterns.</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.header("⚙️ How the System Works")

    step1, step2, step3, step4 = st.columns(4)
    workflow = [
        ("1", "Load Data", "Patient records are loaded from the dataset."),
        ("2", "Preprocess", "Symptoms and categorical values are converted into numerical features."),
        ("3", "Detect", "DBSCAN identifies normal and unusual data patterns."),
        ("4", "Analyze", "A new patient can be compared with learned data patterns.")
    ]

    for column, (number, title, description) in zip([step1, step2, step3, step4], workflow):
        with column:
            st.markdown(
                f"""<div class="workflow-card">
                    <div class="workflow-number">{number}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-text">{description}</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("⚠️ This system identifies unusual data patterns. It does not provide a medical diagnosis.")


elif page == "🔍 Analyze Patient":



    st.title(

        "🔍 Analyze Patient"

    )



    st.write(

        "Enter patient information to analyze "

        "the patient's pattern."

    )



    st.markdown("---")





    # =====================================================

    # PATIENT INFORMATION

    # =====================================================



    st.header(

        "👤 Patient Information"

    )



    col1, col2, col3 = st.columns(3)



    with col1:



        patient_id = st.text_input(

            "Patient ID",

            placeholder="Example: P701"

        )



    with col2:



        age = st.number_input(

            "Age",

            min_value=0,

            max_value=120,

            value=30

        )



    with col3:



        gender = st.selectbox(

            "Gender",

            ["Male", "Female"]

        )





    # =====================================================

    # SYMPTOMS

    # =====================================================



    st.header(

        "🩺 Symptoms"

    )



    symptom_col, guide_col = st.columns([3, 1])



    with symptom_col:



        col1, col2 = st.columns(2)



        with col1:



            fatigue = st.slider(

                "Fatigue",

                0,

                10,

                5

            )



            fever = st.slider(

                "Fever",

                0,

                10,

                5

            )



            weakness = st.slider(

                "Weakness",

                0,

                10,

                5

            )



        with col2:



            seizures = st.slider(

                "Seizures",

                0,

                10,

                5

            )



            pain = st.slider(

                "Pain",

                0,

                10,

                5

            )



            symptom_duration = st.number_input(

                "Symptom Duration (Days)",

                min_value=0,

                max_value=1000,

                value=30

            )



    with guide_col:



        st.info(

            "### Meaning Guide\n\n"

            "**0** — None\n\n"

            "**1–2** — Very Mild\n\n"

            "**3–4** — Moderate\n\n"

            "**5–7** — Severe\n\n"

            "**8–10** — Very Severe"

        )



        st.caption(

            "This guide is for the project interface "

            "and is not a clinical severity scale."

        )





    # =====================================================

    # MEDICAL HISTORY

    # =====================================================



    st.header(

        "🏥 Medical History"

    )



    col1, col2 = st.columns(2)



    with col1:



        family_history = st.selectbox(

            "Family History",

            ["No", "Yes"]

        )



    with col2:



        hospital_visits = st.number_input(

            "Hospital Visits Last Year",

            min_value=0,

            max_value=100,

            value=2

        )





    # =====================================================

    # VALIDATION + ANALYSIS

    # =====================================================



    st.markdown("---")



    if st.button(

        "🔎 Analyze Patient",

        use_container_width=True

    ):



        if not patient_id.strip():



            st.error(

                "Please enter a Patient ID before analysis."

            )



        elif age <= 0:



            st.error(

                "Please enter a valid age."

            )



        else:



            gender_value = (

                0 if gender == "Male" else 1

            )



            family_history_value = (

                0 if family_history == "No" else 1

            )





            new_patient = pd.DataFrame(

                [[

                    age,

                    fatigue,

                    fever,

                    weakness,

                    seizures,

                    pain,

                    gender_value,

                    symptom_duration,

                    family_history_value,

                    hospital_visits

                ]],

                columns=features

            )





            new_patient_scaled = scaler.transform(

                new_patient

            )





            # =================================================

            # DBSCAN CORE SAMPLE COMPARISON

            # =================================================



            if len(

                dbscan.core_sample_indices_

            ) > 0:



                core_samples = X_scaled[

                    dbscan.core_sample_indices_

                ]



                distances = (

                    (

                        core_samples -

                        new_patient_scaled

                    ) ** 2

                ).sum(axis=1)



                minimum_distance = (

                    distances.min() ** 0.5

                )



            else:



                minimum_distance = float("inf")





            if minimum_distance > dbscan.eps:



                is_unusual = True



            else:



                is_unusual = False





            # =================================================

            # PATIENT SUMMARY

            # =================================================



            st.markdown("---")



            st.header(

                "📋 Patient Summary"

            )



            col1, col2, col3, col4 = st.columns(4)



            with col1:



                st.metric(

                    "Patient ID",

                    patient_id

                )



            with col2:



                st.metric(

                    "Age",

                    age

                )



            with col3:



                st.metric(

                    "Gender",

                    gender

                )



            with col4:



                st.metric(

                    "Hospital Visits",

                    hospital_visits

                )





            # =================================================

            # RESULT

            # =================================================



            st.header(

                "🧠 Detection Result"

            )



            if is_unusual:



                st.markdown(

                    """

                    <div class="unusual-result">

                        <div class="result-title">

                            🔴 UNUSUAL CASE

                        </div>

                        <div class="result-text">

                            The patient's pattern is different

                            from the common patterns identified

                            in the dataset.

                        </div>

                    </div>

                    """,

                    unsafe_allow_html=True

                )



            else:



                st.markdown(

                    """

                    <div class="normal-result">

                        <div class="result-title">

                            🟢 NORMAL PATTERN

                        </div>

                        <div class="result-text">

                            The patient's pattern is similar

                            to the common patterns identified

                            in the dataset.

                        </div>

                    </div>

                    """,

                    unsafe_allow_html=True

                )





            col1, col2 = st.columns(2)



            with col1:



                st.metric(

                    "Pattern Distance",

                    f"{minimum_distance:.3f}"

                )



            with col2:



                if is_unusual:



                    st.metric(

                        "Detection Status",

                        "Unusual"

                    )



                else:



                    st.metric(

                        "Detection Status",

                        "Normal"

                    )





            # =================================================

            # FEATURE SUMMARY

            # =================================================



            st.header(

                "📊 Feature Summary"

            )



            average_values = {



                "Age":

                    data["Age"].mean(),



                "Fatigue":

                    data["Fatigue"].mean(),



                "Fever":

                    data["Fever"].mean(),



                "Weakness":

                    data["Weakness"].mean(),



                "Seizures":

                    data["Seizures"].mean(),



                "Pain":

                    data["Pain"].mean(),



                "Symptom Duration":

                    data[

                        "Symptom_Duration_Days"

                    ].mean(),



                "Hospital Visits":

                    data[

                        "Hospital_Visits_Last_Year"

                    ].mean()

            }





            patient_values = {



                "Age":

                    age,



                "Fatigue":

                    fatigue,



                "Fever":

                    fever,



                "Weakness":

                    weakness,



                "Seizures":

                    seizures,



                "Pain":

                    pain,



                "Symptom Duration":

                    symptom_duration,



                "Hospital Visits":

                    hospital_visits

            }





            feature_rows = []





            for feature_name in patient_values:



                patient_value = patient_values[

                    feature_name

                ]



                average_value = average_values[

                    feature_name

                ]





                if patient_value > (

                    average_value * 1.25

                ):



                    level = "High"



                elif patient_value < (

                    average_value * 0.75

                ):



                    level = "Low"



                else:



                    level = "Normal"





                feature_rows.append({



                    "Feature":

                        feature_name,



                    "Patient Value":

                        round(

                            patient_value,

                            2

                        ),



                    "Dataset Average":

                        round(

                            average_value,

                            2

                        ),



                    "Level":

                        level

                })





            feature_table = pd.DataFrame(

                feature_rows

            )





            st.dataframe(

                feature_table,

                use_container_width=True,

                hide_index=True

            )





            # =================================================

            # PATTERN INDICATORS

            # =================================================



            st.header(

                "🔎 Pattern Indicators"

            )



            st.caption(

                "These indicators describe how the patient's input values "

                "compare with dataset averages. They are not medical findings."

            )



            indicator_rows = []



            for symptom_name in [

                "Fatigue",

                "Fever",

                "Weakness",

                "Seizures",

                "Pain"

            ]:



                patient_value = patient_values[symptom_name]

                average_value = average_values[symptom_name]



                if patient_value > average_value * 1.25:

                    comparison = "High"

                elif patient_value < average_value * 0.75:

                    comparison = "Low"

                else:

                    comparison = "Normal"



                indicator_rows.append({

                    "Symptom": symptom_name,

                    "Patient Score": int(patient_value),

                    "Meaning": severity_meaning(patient_value),

                    "Compared With Dataset": comparison

                })



            indicator_table = pd.DataFrame(indicator_rows)



            st.dataframe(

                indicator_table,

                use_container_width=True,

                hide_index=True

            )





            # =================================================

            # REPORT

            # =================================================



            st.header(

                "📄 Patient Analysis Report"

            )





            if is_unusual:



                status_text = "UNUSUAL CASE"



            else:



                status_text = "NORMAL PATTERN"





            report = f"""

RARE DISEASE CASE IDENTIFICATION REPORT

========================================



Patient ID: {patient_id}

Age: {age}

Gender: {gender}



Fatigue: {fatigue}

Fever: {fever}

Weakness: {weakness}

Seizures: {seizures}

Pain: {pain}



Symptom Duration: {symptom_duration} days

Family History: {family_history}

Hospital Visits Last Year: {hospital_visits}



Detection Status: {status_text}



Pattern Distance: {minimum_distance:.3f}



Interpretation:

The DBSCAN-based system identified the

patient's pattern as {status_text.lower()}.



This result represents a machine learning

pattern detection result and is not a

medical diagnosis.

"""





            st.text_area(

                "Report Preview",

                report,

                height=330

            )





            st.download_button(

                "📥 Download Analysis Report",

                data=report,

                file_name="patient_analysis_report.txt",

                mime="text/plain",

                use_container_width=True

            )





            st.warning(

                "⚠️ This result is for educational "

                "and research purposes only and "

                "should not be considered a medical diagnosis."

            )





# =========================================================

# 3. ADD PATIENT

# =========================================================



elif page == "➕ Add Patient":



    st.title(

        "➕ Add Patient"

    )



    st.write(

        "Add a new patient record to the dataset."

    )



    st.markdown("---")





    # =====================================================

    # PATIENT INFORMATION

    # =====================================================



    st.header(

        "👤 Patient Information"

    )



    col1, col2, col3 = st.columns(3)



    with col1:



        new_patient_id = st.text_input(

            "Patient ID",

            placeholder="Example: P701"

        )



    with col2:



        new_age = st.number_input(

            "Age",

            min_value=0,

            max_value=120,

            value=30

        )



    with col3:



        new_gender = st.selectbox(

            "Gender",

            ["Male", "Female"]

        )





    # =====================================================

    # SYMPTOMS

    # =====================================================



    st.header(

        "🩺 Symptoms"

    )



    symptom_col, guide_col = st.columns([3, 1])



    with symptom_col:



        col1, col2 = st.columns(2)



        with col1:



            new_fatigue = st.slider(

                "Fatigue",

                0,

                10,

                5

            )



            new_fever = st.slider(

                "Fever",

                0,

                10,

                5

            )



            new_weakness = st.slider(

                "Weakness",

                0,

                10,

                5

            )



        with col2:



            new_seizures = st.slider(

                "Seizures",

                0,

                10,

                5

            )



            new_pain = st.slider(

                "Pain",

                0,

                10,

                5

            )



            new_duration = st.number_input(

                "Symptom Duration (Days)",

                min_value=0,

                max_value=1000,

                value=30

            )



    with guide_col:



        st.info(

            "### Meaning Guide\n\n"

            "**0** — None\n\n"

            "**1–2** — Very Mild\n\n"

            "**3–4** — Moderate\n\n"

            "**5–7** — Severe\n\n"

            "**8–10** — Very Severe"

        )



        st.caption(

            "This guide is for the project interface "

            "and is not a clinical severity scale."

        )



    # =====================================================

    # MEDICAL HISTORY

    # =====================================================



    st.header(

        "🏥 Medical History"

    )



    col1, col2 = st.columns(2)



    with col1:



        new_family_history = st.selectbox(

            "Family History",

            ["No", "Yes"]

        )



    with col2:



        new_hospital_visits = st.number_input(

            "Hospital Visits Last Year",

            min_value=0,

            max_value=100,

            value=2

        )





    # =====================================================

    # CASE INFORMATION

    # =====================================================



    st.header(

        "🏷️ Case Information"

    )



    new_label = st.selectbox(

        "Case Label",

        ["Rare", "Common"]

    )





    # =====================================================

    # ADD PATIENT

    # =====================================================



    st.markdown("---")



    if st.button(

        "➕ Add Patient",

        use_container_width=True

    ):



        if not new_patient_id.strip():



            st.error(

                "Please enter a Patient ID."

            )



        elif new_age <= 0:



            st.error(

                "Please enter a valid age."

            )



        else:



            existing_ids = (

                data["PatientID"]

                .astype(str)

                .tolist()

            )





            if new_patient_id.strip() in existing_ids:



                st.error(

                    "Patient ID already exists."

                )



            else:



                new_record = pd.DataFrame(

                    [[

                        new_patient_id.strip(),

                        new_age,

                        f"{new_fatigue} (Severity)",

                        f"{new_fever} (Severity)",

                        f"{new_weakness} (Severity)",

                        f"{new_seizures} (Severity)",

                        f"{new_pain} (Severity)",

                        new_gender,

                        new_duration,

                        new_family_history,

                        new_hospital_visits,

                        new_label

                    ]],

                    columns=[

                        "PatientID",

                        "Age",

                        "Fatigue",

                        "Fever",

                        "Weakness",

                        "Seizures",

                        "Pain",

                        "Gender",

                        "Symptom_Duration_Days",

                        "Family_History",

                        "Hospital_Visits_Last_Year",

                        "Label"

                    ]

                )





                current_data = pd.read_csv(

                    "data.csv",

                    sep="\t"

                )





                updated_data = pd.concat(

                    [

                        current_data,

                        new_record

                    ],

                    ignore_index=True

                )





                updated_data.to_csv(

                    "data.csv",

                    sep="\t",

                    index=False

                )





                st.cache_data.clear()





                st.success(

                    f"Patient {new_patient_id} "

                    "was added successfully!"

                )





                st.info(

                    "The model will use the updated "

                    "dataset after the application refreshes."

                )





                st.rerun()





# =========================================================

# 4. PATIENT DETAILS

# =========================================================



elif page == "👤 Patient Details":



    st.title(

        "👤 Patient Details"

    )



    st.write(

        "View detailed information about a patient."

    )



    st.markdown("---")





    patient_ids = (

        data["PatientID"]

        .astype(str)

        .tolist()

    )





    selected_patient = st.selectbox(

        "Select Patient",

        patient_ids

    )





    patient = data[

        data["PatientID"].astype(str)

        == selected_patient

    ].iloc[0]





    # =====================================================

    # BASIC INFORMATION

    # =====================================================



    st.header(

        "👤 Patient Information"

    )



    col1, col2, col3 = st.columns(3)





    with col1:



        st.metric(

            "Patient ID",

            str(patient["PatientID"])

        )





    with col2:



        st.metric(

            "Age",

            int(patient["Age"])

        )





    with col3:



        gender_display = (

            "Male"

            if patient["Gender"] == 0

            else "Female"

        )



        st.metric(

            "Gender",

            gender_display

        )





    # =====================================================

    # SYMPTOMS

    # =====================================================



    st.header(

        "🩺 Symptoms"

    )





    symptom_table = pd.DataFrame({



        "Symptom": [

            "Fatigue",

            "Fever",

            "Weakness",

            "Seizures",

            "Pain",

            "Symptom Duration"

        ],



        "Value": [

            patient["Fatigue"],

            patient["Fever"],

            patient["Weakness"],

            patient["Seizures"],

            patient["Pain"],

            patient["Symptom_Duration_Days"]

        ],



        "Meaning": [

            severity_meaning(patient["Fatigue"]),

            severity_meaning(patient["Fever"]),

            severity_meaning(patient["Weakness"]),

            severity_meaning(patient["Seizures"]),

            severity_meaning(patient["Pain"]),

            "Days"

        ]



    })





    st.dataframe(

        symptom_table,

        use_container_width=True,

        hide_index=True

    )



    st.info(

        "### Meaning Guide\n\n"

        "**0** — None  |  "

        "**1–2** — Very Mild  |  "

        "**3–4** — Moderate  |  "

        "**5–7** — Severe  |  "

        "**8–10** — Very Severe\n\n"

        "This guide is for the project interface and is "

        "not a clinical severity scale."

    )





    # =====================================================

    st.subheader("📈 Symptom Profile")

    visual_symptoms = [
        ("Fatigue", patient["Fatigue"]),
        ("Fever", patient["Fever"]),
        ("Weakness", patient["Weakness"]),
        ("Seizures", patient["Seizures"]),
        ("Pain", patient["Pain"])
    ]

    visual_col1, visual_col2 = st.columns(2)

    for index, (symptom_name, symptom_value) in enumerate(visual_symptoms):
        target_col = visual_col1 if index < 3 else visual_col2

        with target_col:
            numeric_value = float(symptom_value)
            percentage = min(max(numeric_value / 10 * 100, 0), 100)

            st.markdown(
                f"""
                <div class="symptom-visual">
                    <div class="symptom-name">{symptom_name}</div>
                    <div class="symptom-bar-bg">
                        <div class="symptom-bar" style="width: {percentage:.0f}%"></div>
                    </div>
                    <div class="symptom-meta">
                        <span>{severity_meaning(numeric_value)}</span>
                        <span>{int(numeric_value)}/10</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # MEDICAL HISTORY

    # =====================================================



    st.header(

        "🏥 Medical History"

    )





    family_display = (

        "No"

        if patient["Family_History"] == 0

        else "Yes"

    )





    col1, col2 = st.columns(2)





    with col1:



        st.info(

            f"**Family History:** {family_display}"

        )





    with col2:



        st.info(

            f"**Hospital Visits Last Year:** "

            f"{int(patient['Hospital_Visits_Last_Year'])}"

        )





    # =====================================================

    # DATASET LABEL

    # =====================================================



    st.header(

        "🏷️ Dataset Label"

    )





    if patient["Label"] == "Rare":



        st.warning(

            "Rare Case"

        )



    else:



        st.success(

            "Common Case"

        )





    st.info(

        "The Rare/Common label is a reference "

        "label in the dataset and is not used "

        "during DBSCAN training."

    )





    # =====================================================

    # DBSCAN PATTERN

    # =====================================================



    st.header(

        "🤖 DBSCAN Pattern"

    )



    patient_cluster = patient["Cluster"]



    if patient_cluster == -1:

        st.error(

            "🔴 Unusual Pattern\n\n"

            "This patient record was identified as an unusual "

            "data pattern by DBSCAN."

        )

    else:

        st.success(

            "🟢 Normal Pattern\n\n"

            "This patient record belongs to the main data pattern "

            "identified by DBSCAN."

        )



    st.caption(

        "DBSCAN pattern detection is a machine learning result "

        "and is not a medical diagnosis."

    )





# =========================================================

# 5. PATIENT DATASET

# =========================================================



elif page == "👥 Patient Dataset":



    st.title(

        "👥 Patient Dataset"

    )



    st.write(

        "View and filter the patient dataset."

    )



    st.markdown("---")





    # =====================================================

    # SUMMARY

    # =====================================================



    st.header(

        "📊 Dataset Summary"

    )





    col1, col2, col3 = st.columns(3)





    with col1:



        st.metric(

            "Total Patients",

            total_patients

        )





    with col2:



        st.metric(

            "Rare Cases",

            rare_cases

        )





    with col3:



        st.metric(

            "Common Cases",

            common_cases

        )





    # =====================================================

    # SEARCH AND FILTER

    # =====================================================



    st.header(

        "🔎 Search & Filter"

    )





    col1, col2 = st.columns(2)





    with col1:



        search_patient = st.text_input(

            "Search Patient ID",

            placeholder="Enter Patient ID"

        )





    with col2:



        case_filter = st.selectbox(

            "Case Type",

            [

                "All Cases",

                "Rare",

                "Common"

            ]

        )





    patient_dataset = data.drop(

        columns=["Cluster"]

    ).copy()





    if search_patient:



        patient_dataset = patient_dataset[

            patient_dataset["PatientID"]

            .astype(str)

            .str.contains(

                search_patient,

                case=False,

                na=False

            )

        ]





    if case_filter != "All Cases":



        patient_dataset = patient_dataset[

            patient_dataset["Label"]

            == case_filter

        ]





    st.dataframe(

        patient_dataset,

        use_container_width=True,

        hide_index=True

    )





    # =====================================================

    # DOWNLOAD

    # =====================================================



    st.markdown("---")



    download_data = data.drop(

        columns=["Cluster"]

    ).copy()





    csv_data = download_data.to_csv(

        sep="\t",

        index=False

    )





    st.download_button(

        label="📥 Download Patient Dataset",

        data=csv_data,

        file_name="rare_disease_patient_dataset.csv",

        mime="text/tab-separated-values",

        use_container_width=True

    )





# =========================================================

# 6. DATASET STATISTICS

# =========================================================



elif page == "📊 Dataset Statistics":



    st.title(

        "📊 Dataset Statistics"

    )



    st.write(

        "Summary statistics of the patient dataset."

    )



    st.markdown("---")





    # =====================================================

    # OVERALL SUMMARY

    # =====================================================



    st.header(

        "📋 Overall Summary"

    )





    col1, col2, col3, col4 = st.columns(4)





    with col1:



        st.metric(

            "Total Patients",

            total_patients

        )





    with col2:



        st.metric(

            "Rare Cases",

            rare_cases

        )





    with col3:



        st.metric(

            "Common Cases",

            common_cases

        )





    with col4:



        st.metric(

            "Unusual Cases",

            unusual_cases

        )





    # =====================================================

    # PATIENT INFORMATION

    # =====================================================



    st.header(

        "👤 Patient Information"

    )





    col1, col2 = st.columns(2)





    with col1:



        st.metric(

            "Average Age",

            f"{data['Age'].mean():.2f}"

        )





    with col2:



        st.metric(

            "Average Symptom Duration",

            f"{data['Symptom_Duration_Days'].mean():.2f} days"

        )





    # =====================================================

    # AVERAGE SYMPTOMS

    # =====================================================



    st.header(

        "🩺 Average Symptom Severity"

    )





    avg_fatigue = data["Fatigue"].mean()

    avg_fever = data["Fever"].mean()

    avg_weakness = data["Weakness"].mean()

    avg_seizures = data["Seizures"].mean()

    avg_pain = data["Pain"].mean()

    avg_visits = data[

        "Hospital_Visits_Last_Year"

    ].mean()





    col1, col2, col3 = st.columns(3)





    with col1:



        st.metric(

            "Fatigue",

            f"{avg_fatigue:.2f}"

        )



        st.metric(

            "Fever",

            f"{avg_fever:.2f}"

        )





    with col2:



        st.metric(

            "Weakness",

            f"{avg_weakness:.2f}"

        )



        st.metric(

            "Seizures",

            f"{avg_seizures:.2f}"

        )





    with col3:



        st.metric(

            "Pain",

            f"{avg_pain:.2f}"

        )



        st.metric(

            "Hospital Visits",

            f"{avg_visits:.2f}"

        )



    st.info(

        "### Symptom Score Meaning Guide\n\n"

        "**0** — None  |  "

        "**1–2** — Very Mild  |  "

        "**3–4** — Moderate  |  "

        "**5–7** — Severe  |  "

        "**8–10** — Very Severe\n\n"

        "The guide describes the 0–10 interface values "

        "used in this academic project. It is not a "

        "clinical severity scale."

    )





    # =====================================================

    # SUMMARY

    # =====================================================



    st.header(

        "📝 Dataset Summary"

    )





    st.write(

        f"The dataset contains {total_patients} "

        f"patient records, including "

        f"{rare_cases} Rare cases and "

        f"{common_cases} Common cases."

    )





    st.write(

        f"DBSCAN identified {unusual_cases} "

        f"records as unusual patterns."

    )





    st.info(

        "These statistics describe the dataset "

        "and are not medical diagnostic measurements."

    )





# =========================================================

# 7. ANALYTICS

# =========================================================



elif page == "📋 Analytics":



    st.title(

        "📋 Analytics"

    )



    st.write(

        "Summary of patterns identified by "

        "the DBSCAN model."

    )



    st.markdown("---")





    # =====================================================

    # CASE SUMMARY

    # =====================================================



    st.header(

        "📊 Case Summary"

    )





    col1, col2, col3 = st.columns(3)





    unusual_rate = (

        unusual_cases / total_patients * 100

        if total_patients > 0

        else 0

    )





    with col1:



        st.metric(

            "Normal Cases",

            normal_cases

        )





    with col2:



        st.metric(

            "Unusual Cases",

            unusual_cases

        )





    with col3:



        st.metric(

            "Unusual Rate",

            f"{unusual_rate:.2f}%"

        )





    # =====================================================

    # UNUSUAL CASES

    # =====================================================



    st.header(

        "🔎 Unusual Cases"

    )





    unusual_data = data[

        data["Cluster"] == -1

    ][

        [

            "PatientID",

            "Age",

            "Label"

        ]

    ].copy()





    if len(unusual_data) > 0:



        st.dataframe(

            unusual_data,

            use_container_width=True,

            hide_index=True

        )



    else:



        st.success(

            "No unusual cases were identified."

        )





    # =====================================================

    # INTERPRETATION

    # =====================================================



    st.header(

        "📝 Interpretation"

    )





    st.write(

        f"DBSCAN identified {unusual_cases} "

        f"patient records as unusual patterns "

        f"and {normal_cases} records as normal patterns "

        f"({unusual_rate:.2f}% of the dataset was identified as unusual)."

    )





    st.info(

        "Unusual cases represent records with "

        "patterns that differ from the main data "

        "distribution. They are not confirmed "

        "medical diagnoses."

    )



# =========================================================

# APPLICATION FOOTER

# =========================================================



st.markdown("---")



st.caption(

    "🧬 Rare Disease Case Identification • "

    "DBSCAN-Based Anomaly Detection • "

    "B.Tech Machine Learning Project"

)



st.caption(

    "⚠️ For educational and research purposes only. "

    "This system is not a medical diagnostic tool."

)
