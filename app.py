import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import os

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="EduPro - Student Learning Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# EDUCATION THEME CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
    /* ===== MAIN BACKGROUND - BOOK THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #f5f0e8 0%, #e8dcc8 50%, #d4c4a8 100%);
        background-image: 
            url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c4a882' fill-opacity='0.15'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    
    /* ===== CONTENT CARDS - PAPER TEXTURE ===== */
    .main > div {
        background: rgba(255, 252, 245, 0.92);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 8px 32px 0 rgba(0, 0, 0, 0.08),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 252, 245, 0.6);
        position: relative;
        overflow: hidden;
    }
    
    .main > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 28px,
                rgba(200, 180, 160, 0.05) 28px,
                rgba(200, 180, 160, 0.05) 29px
            );
        pointer-events: none;
        z-index: 0;
    }
    
    /* ===== SIDEBAR - DARK ACADEMIC THEME ===== */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c1810 0%, #4a2c1a 30%, #6b3f2a 100%);
        padding: 2rem 1rem;
        border-radius: 0 24px 24px 0;
        border-right: 3px solid #c4a882;
        box-shadow: 4px 0 20px 0 rgba(0, 0, 0, 0.2);
    }
    
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stSlider label {
        color: #f5f0e8 !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stSelectbox {
        background: rgba(255, 252, 245, 0.1);
        border-radius: 12px;
        padding: 0.25rem;
    }
    
    /* ===== HEADERS - EDUCATION STYLE ===== */
    h1 {
        font-family: 'Georgia', serif !important;
        background: linear-gradient(135deg, #2c1810 0%, #6b3f2a 50%, #8b5a3c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
        position: relative;
        display: inline-block;
    }
    
    h1::after {
        content: '📖';
        font-size: 2.5rem;
        -webkit-text-fill-color: initial;
        margin-left: 0.5rem;
        opacity: 0.7;
    }
    
    h2 {
        font-family: 'Georgia', serif !important;
        color: #2c1810 !important;
        font-weight: 600 !important;
        border-bottom: 3px solid #c4a882;
        padding-bottom: 0.5rem;
        position: relative;
    }
    
    h2::before {
        content: '✧ ';
        color: #8b5a3c;
    }
    
    h3 {
        font-family: 'Georgia', serif !important;
        color: #4a2c1a !important;
        font-weight: 600 !important;
    }
    
    /* ===== METRIC CARDS - EDUCATION COLORS ===== */
    .stMetric {
        background: linear-gradient(135deg, #2c1810 0%, #6b3f2a 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: #f5f0e8 !important;
        box-shadow: 
            0 4px 15px 0 rgba(44, 24, 16, 0.2),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(196, 168, 130, 0.3);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px 0 rgba(44, 24, 16, 0.3);
    }
    
    .stMetric::before {
        content: '📘';
        position: absolute;
        right: 1rem;
        top: 0.5rem;
        font-size: 2rem;
        opacity: 0.15;
    }
    
    .stMetric label {
        color: rgba(245, 240, 232, 0.85) !important;
        font-weight: 500 !important;
    }
    
    .stMetric .stMetricValue {
        color: #f5f0e8 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    /* ===== BUTTONS - QUILL PEN THEME ===== */
    .stButton > button {
        background: linear-gradient(135deg, #2c1810 0%, #6b3f2a 100%) !important;
        color: #f5f0e8 !important;
        border: 1px solid #c4a882 !important;
        border-radius: 25px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Georgia', serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px 0 rgba(44, 24, 16, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 6px 20px 0 rgba(44, 24, 16, 0.3);
        background: linear-gradient(135deg, #4a2c1a 0%, #8b5a3c 100%) !important;
    }
    
    /* ===== TABS - BOOKMARK THEME ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(245, 240, 232, 0.7);
        border-radius: 30px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(196, 168, 130, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 25px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-family: 'Georgia', serif !important;
        transition: all 0.3s ease;
        color: #4a2c1a;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(196, 168, 130, 0.15);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2c1810 0%, #6b3f2a 100%) !important;
        color: #f5f0e8 !important;
        box-shadow: 0 2px 10px 0 rgba(44, 24, 16, 0.2);
    }
    
    /* ===== EXPANDERS - SCROLL THEME ===== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f0e8 0%, #e8dcc8 100%) !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-family: 'Georgia', serif !important;
        color: #2c1810 !important;
        border: 1px solid #c4a882 !important;
        transition: all 0.3s ease !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        transform: translateX(8px);
        box-shadow: 0 4px 12px 0 rgba(44, 24, 16, 0.15);
        background: linear-gradient(135deg, #e8dcc8 0%, #d4c4a8 100%) !important;
    }
    
    .streamlit-expanderHeader::before {
        content: '📜 ';
    }
    
    /* ===== DATA FRAMES - LEDGER STYLE ===== */
    .stDataFrame {
        background: #faf8f5;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e8dcc8;
    }
    
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #2c1810, #6b3f2a) !important;
        color: #f5f0e8 !important;
        font-family: 'Georgia', serif !important;
        padding: 0.75rem !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .stDataFrame tbody tr:hover {
        background: rgba(196, 168, 130, 0.1) !important;
    }
    
    /* ===== ALERT MESSAGES ===== */
    .stAlert {
        border-radius: 16px !important;
        border: 1px solid #c4a882 !important;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.05);
        background: #faf8f5 !important;
    }
    
    .stAlert .stMarkdown {
        color: #2c1810 !important;
    }
    
    /* ===== INFO BOXES - MARGIN NOTE STYLE ===== */
    .stInfo {
        background: linear-gradient(135deg, #f5f0e8 0%, #e8dcc8 100%) !important;
        border-left: 5px solid #8b5a3c !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* ===== SIDEBAR HEADER ===== */
    .css-1d391kg h2 {
        color: #f5f0e8 !important;
        border-bottom: 2px solid #c4a882;
        text-align: center;
        padding-bottom: 1rem;
        font-family: 'Georgia', serif !important;
    }
    
    .css-1d391kg h2::before {
        content: '🏛️ ';
        -webkit-text-fill-color: initial;
    }
    
    /* ===== CUSTOM COURSE CARD - BOOK SPINE THEME ===== */
    .course-card {
        background: #faf8f5;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-left: 6px solid #8b5a3c;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border: 1px solid #e8dcc8;
        border-left-width: 6px;
        position: relative;
    }
    
    .course-card:hover {
        transform: translateX(12px) scale(1.01);
        box-shadow: 0 6px 20px 0 rgba(44, 24, 16, 0.12);
        background: #ffffff;
    }
    
    .course-card::after {
        content: '📘';
        position: absolute;
        right: 1rem;
        top: 0.5rem;
        font-size: 1.5rem;
        opacity: 0.2;
    }
    
    /* ===== BADGES - ACADEMIC SEAL STYLE ===== */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'Georgia', serif !important;
        margin: 0.15rem;
        border: 1px solid rgba(255,255,255,0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-beginner {
        background: linear-gradient(135deg, #1a7a3a, #2a9f5a);
        color: white;
        box-shadow: 0 2px 8px 0 rgba(26, 122, 58, 0.3);
    }
    
    .badge-intermediate {
        background: linear-gradient(135deg, #b8751a, #e89f2a);
        color: white;
        box-shadow: 0 2px 8px 0 rgba(184, 117, 26, 0.3);
    }
    
    .badge-advanced {
        background: linear-gradient(135deg, #8b1a2a, #bf2a3a);
        color: white;
        box-shadow: 0 2px 8px 0 rgba(139, 26, 42, 0.3);
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        color: #8b7a6a;
        font-family: 'Georgia', serif !important;
        font-size: 0.95rem;
        border-top: 2px solid #e8dcc8;
        margin-top: 2rem;
    }
    
    .footer::before {
        content: '✦ ';
        color: #8b5a3c;
    }
    
    .footer::after {
        content: ' ✦';
        color: #8b5a3c;
    }
    
    /* ===== SLIDER STYLING ===== */
    .stSlider .stSliderTrack {
        background: #e8dcc8 !important;
    }
    
    .stSlider .stSliderThumb {
        background: #8b5a3c !important;
        border: 2px solid #f5f0e8 !important;
    }
    
    /* ===== SELECT BOX ===== */
    .stSelectbox select {
        background: #faf8f5 !important;
        border: 1px solid #c4a882 !important;
        border-radius: 12px !important;
        color: #2c1810 !important;
        font-family: 'Georgia', serif !important;
    }
    
    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot {
        background: #faf8f5;
        border-radius: 16px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e8dcc8;
    }
    
    /* ===== SECTION HEADER ===== */
    .section-header {
        background: linear-gradient(135deg, #f5f0e8 0%, #e8dcc8 100%);
        padding: 0.75rem 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #8b5a3c;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Georgia', serif;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.05);
    }
    
    .section-header h2, .section-header h3 {
        margin: 0;
        border-bottom: none;
    }
    
    .section-header h2::before {
        content: '';
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 1. LOAD USER DATA - FIXED FILE PATH
# ----------------------------
@st.cache_data
def load_users():
    """Load all users from CSV file with proper path handling"""
    # Use raw string or forward slashes to avoid escape sequence issues
    # Option 1: Try current directory
    csv_file = "Users.csv"
    
    # Option 2: Try the specific path with raw string (r"...") or forward slashes
    csv_file_path = r"C:\Users\gokul\Downloads\EduPro Online Platform.xlsx - Users.csv"
    
    # Try multiple locations
    possible_paths = [
        "Users.csv",
        r"C:\Users\gokul\Downloads\EduPro Online Platform.xlsx - Users.csv",
        "C:/Users/gokul/Downloads/EduPro Online Platform.xlsx - Users.csv",
        os.path.join(os.path.dirname(__file__), "Users.csv")
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Loaded {len(df)} users from: {os.path.basename(path)}")
                return df
        except Exception as e:
            continue
    
    # If no file found, generate sample data
    st.warning("⚠️ CSV file not found. Generating 3000 sample users for demonstration.")
    np.random.seed(42)
    user_ids = [f'U{i:05d}' for i in range(1, 3001)]
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'James', 'Emma', 'Robert', 'Olivia',
                   'William', 'Sophia', 'Richard', 'Isabella', 'Joseph', 'Mia', 'Thomas', 'Charlotte', 'Charles', 'Amelia',
                   'Christopher', 'Harper', 'Daniel', 'Evelyn', 'Matthew', 'Abigail', 'Anthony', 'Emily', 'Mark', 'Elizabeth',
                   'Donald', 'Sofia', 'Steven', 'Avery', 'Paul', 'Ella', 'Andrew', 'Madison', 'Joshua', 'Scarlett',
                   'Kenneth', 'Victoria', 'Kevin', 'Aria', 'Brian', 'Grace', 'George', 'Chloe', 'Timothy', 'Camila']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                  'Hernandez', 'Lopez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee',
                  'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
                  'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green']
    
    users = []
    for i, user_id in enumerate(user_ids):
        first = np.random.choice(first_names)
        last = np.random.choice(last_names)
        username = f"{first.lower()}{np.random.randint(10, 99)}"
        age = np.random.randint(15, 36)
        gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
        email = f"{username}@email.com"
        users.append({
            'UserID': user_id,
            'UserName': username,
            'Age': age,
            'Gender': gender,
            'Email': email
        })
    
    df = pd.DataFrame(users)
    st.info(f"📊 Generated {len(df)} sample users for demonstration")
    return df

# ----------------------------
# 2. DATA GENERATION (SYNTHETIC COURSES & TRANSACTIONS)
# ----------------------------
@st.cache_data
def generate_courses():
    """Generate synthetic courses"""
    categories = ['Mathematics', 'Science', 'Literature', 'History', 'Languages', 'Arts', 'Technology']
    levels = ['Beginner', 'Intermediate', 'Advanced']
    types = ['Lecture', 'Interactive', 'Reading']
    
    courses = []
    for i in range(1, 61):
        cat = np.random.choice(categories)
        lev = np.random.choice(levels, p=[0.4, 0.4, 0.2])
        typ = np.random.choice(types, p=[0.5, 0.3, 0.2])
        rating = round(np.random.uniform(3.0, 5.0), 1)
        price = round(np.random.uniform(19.99, 99.99), 2)
        courses.append({
            'CourseID': f'C{i:04d}',
            'CourseCategory': cat,
            'CourseType': typ,
            'CourseLevel': lev,
            'CourseRating': rating,
            'Price': price
        })
    return pd.DataFrame(courses)

@st.cache_data
def generate_transactions(users_df, courses_df):
    """Generate synthetic transactions for all users"""
    transactions = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 1)
    
    for _, user in users_df.iterrows():
        user_id = user['UserID']
        age = user['Age']
        gender = user['Gender']
        
        # Determine number of courses based on age
        if age < 20:
            n_courses = np.random.randint(2, 10)
        elif age < 30:
            n_courses = np.random.randint(4, 15)
        else:
            n_courses = np.random.randint(3, 12)
        
        # Bias toward certain categories based on gender
        categories = ['Mathematics', 'Science', 'Literature', 'History', 'Languages', 'Arts', 'Technology']
        if gender == 'Male':
            preferred = np.random.choice(['Mathematics', 'Science', 'Technology'], p=[0.4, 0.3, 0.3])
        else:
            preferred = np.random.choice(['Literature', 'Languages', 'Arts', 'History'], p=[0.3, 0.3, 0.2, 0.2])
        
        # Choose courses
        selected_courses = []
        pref_courses = courses_df[courses_df['CourseCategory'] == preferred]['CourseID'].tolist()
        other_courses = courses_df[courses_df['CourseCategory'] != preferred]['CourseID'].tolist()
        
        for _ in range(n_courses):
            if np.random.rand() < 0.6 and pref_courses:
                course_id = np.random.choice(pref_courses)
            else:
                course_id = np.random.choice(other_courses) if other_courses else np.random.choice(courses_df['CourseID'])
            
            # Avoid duplicate enrollment per user
            while course_id in selected_courses:
                course_id = np.random.choice(courses_df['CourseID'])
            selected_courses.append(course_id)
        
        # Create transaction records
        for cid in selected_courses:
            days = (end_date - start_date).days
            date = start_date + timedelta(days=np.random.randint(0, days))
            transactions.append({
                'UserID': user_id,
                'CourseID': cid,
                'TransactionDate': date.strftime('%Y-%m-%d'),
                'Amount': None
            })
    
    transactions_df = pd.DataFrame(transactions)
    # Merge with courses to get price
    transactions_df = transactions_df.merge(courses_df[['CourseID', 'Price']], on='CourseID', how='left')
    transactions_df['Amount'] = transactions_df['Price']
    transactions_df.drop('Price', axis=1, inplace=True)
    
    return transactions_df

# ----------------------------
# 3. FEATURE ENGINEERING
# ----------------------------
def create_learner_features(users_df, transactions_df, courses_df):
    categories = ['Mathematics', 'Science', 'Literature', 'History', 'Languages', 'Arts', 'Technology']
    levels = ['Beginner', 'Intermediate', 'Advanced']
    
    # Aggregate transactions per user
    trans_agg = transactions_df.groupby('UserID').agg(
        total_courses=('CourseID', 'count'),
        total_spent=('Amount', 'sum')
    ).reset_index()
    
    # Per category count
    trans_cat = transactions_df.merge(courses_df[['CourseID', 'CourseCategory']], on='CourseID')
    cat_pivot = trans_cat.pivot_table(index='UserID', columns='CourseCategory', values='CourseID', aggfunc='count', fill_value=0)
    
    # Merge features
    features = users_df.merge(trans_agg, on='UserID', how='left').fillna(0)
    features = features.merge(cat_pivot, on='UserID', how='left').fillna(0)
    
    # Feature engineering
    cat_cols = [c for c in features.columns if c in categories]
    features['diversity_score'] = (features[cat_cols] > 0).sum(axis=1)
    features['preferred_category'] = features[cat_cols].idxmax(axis=1)
    features['preferred_category'] = features['preferred_category'].where(features[cat_cols].max(axis=1) > 0, 'None')
    
    # Preferred level
    trans_level = transactions_df.merge(courses_df[['CourseID', 'CourseLevel']], on='CourseID')
    level_counts = trans_level.groupby(['UserID', 'CourseLevel']).size().unstack(fill_value=0)
    for lv in levels:
        if lv not in level_counts.columns:
            level_counts[lv] = 0
    features = features.merge(level_counts, on='UserID', how='left').fillna(0)
    features['preferred_level'] = features[levels].idxmax(axis=1)
    features['preferred_level'] = features['preferred_level'].where(features[levels].max(axis=1) > 0, 'None')
    
    # Average rating
    trans_rating = transactions_df.merge(courses_df[['CourseID', 'CourseRating']], on='CourseID')
    avg_rating = trans_rating.groupby('UserID')['CourseRating'].mean().reset_index(name='avg_rating_enrolled')
    features = features.merge(avg_rating, on='UserID', how='left').fillna(0)
    
    # Depth index
    features['depth_index'] = features['Advanced'] / (features['Beginner'] + features['Intermediate'] + 1)
    
    # Average spending per course
    features['avg_spent_per_course'] = features['total_spent'] / features['total_courses'].replace(0, 1)
    
    # Features for clustering
    features_for_clustering = features[['total_courses', 'diversity_score', 'avg_rating_enrolled',
                                        'depth_index', 'avg_spent_per_course']].copy()
    
    return features, features_for_clustering

# ----------------------------
# 4. CLUSTERING
# ----------------------------
def perform_clustering(features_cluster, n_clusters=4):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    silhouette = silhouette_score(scaled, clusters)
    return clusters, scaled, kmeans, scaler, silhouette

# ----------------------------
# 5. RECOMMENDATION ENGINE
# ----------------------------
def get_recommendations(user_id, users_df, transactions_df, courses_df, clusters, features, n_recommend=5):
    user_idx = users_df[users_df['UserID'] == user_id].index[0]
    cluster = clusters[user_idx]
    cluster_users = users_df.iloc[np.where(clusters == cluster)[0]]['UserID'].tolist()
    cluster_trans = transactions_df[transactions_df['UserID'].isin(cluster_users)]
    
    if len(cluster_trans) == 0:
        return pd.DataFrame()
    
    course_pop = cluster_trans.groupby('CourseID').agg(
        popularity=('CourseID', 'count'),
        avg_rating=('CourseID', lambda x: courses_df[courses_df['CourseID'].isin(x)]['CourseRating'].mean())
    ).reset_index()
    
    course_pop = course_pop.merge(courses_df, on='CourseID')
    user_courses = transactions_df[transactions_df['UserID'] == user_id]['CourseID'].tolist()
    course_pop = course_pop[~course_pop['CourseID'].isin(user_courses)]
    
    if len(course_pop) > 0:
        course_pop['score'] = (course_pop['popularity'] * 0.6 + course_pop['avg_rating'] * 0.4)
        recommendations = course_pop.sort_values('score', ascending=False).head(n_recommend)
        return recommendations[['CourseID', 'CourseCategory', 'CourseType', 'CourseLevel', 'CourseRating', 'popularity', 'avg_rating']]
    else:
        return pd.DataFrame()

# ----------------------------
# 6. STREAMLIT APP
# ----------------------------
def main():
    # Custom title with education theme
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1.5rem 0;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 1rem;">
            <span style="font-size: 4rem;">🏛️</span>
            <div>
                <h1 style="font-size: 3.8rem; margin: 0; display: inline-block;">
                    EduPro Academy
                </h1>
            </div>
            <span style="font-size: 4rem;">📚</span>
        </div>
        <p style="font-size: 1.3rem; color: #6b5a4a; margin-top: 0.25rem; font-family: 'Georgia', serif;">
            <em>Student Learning Intelligence & Personalized Course Discovery</em>
        </p>
        <div style="width: 300px; height: 3px; background: linear-gradient(90deg, #2c1810, #c4a882, #2c1810); margin: 0.5rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("📖 Loading academic records and generating learning materials..."):
        users_df = load_users()
        courses_df = generate_courses()
        transactions_df = generate_transactions(users_df, courses_df)
        features, features_cluster = create_learner_features(users_df, transactions_df, courses_df)
    
    # Academic-style success message
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2c1810 0%, #6b3f2a 100%);
                padding: 1.25rem 2rem;
                border-radius: 16px;
                color: #f5f0e8;
                box-shadow: 0 4px 15px 0 rgba(44, 24, 16, 0.2);
                margin: 0.5rem 0 1.5rem 0;
                border: 1px solid #c4a882;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;">
        <span>📘 <strong>{len(users_df)}</strong> Scholars enrolled</span>
        <span>📚 <strong>{len(courses_df)}</strong> Courses available</span>
        <span>📝 <strong>{len(transactions_df)}</strong> Enrollments recorded</span>
        <span>🎯 <strong>{len(features['preferred_category'].unique())}</strong> Subjects explored</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with academic theme
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 3rem;">📜</div>
        <h2 style="color: #f5f0e8; margin: 0; font-family: 'Georgia', serif;">Study Settings</h2>
        <div style="width: 50px; height: 2px; background: #c4a882; margin: 0.5rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("🎯 Segmentation")
    n_clusters = st.sidebar.slider(
        "Number of learner groups",
        2, 8, 4,
        help="More clusters = finer segmentation"
    )
    
    # Perform clustering
    with st.spinner("🧠 Analyzing learning patterns..."):
        clusters, scaled, kmeans, scaler, silhouette = perform_clustering(features_cluster, n_clusters)
        features['cluster'] = clusters
    
    st.sidebar.markdown("---")
    st.sidebar.header("👤 Scholar Explorer")
    user_ids = users_df['UserID'].tolist()
    selected_user = st.sidebar.selectbox("Select a learner", user_ids)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,252,245,0.08); 
                padding: 1rem; 
                border-radius: 12px;
                border: 1px solid #c4a882;
                text-align: center;">
        <div style="font-size: 0.8rem; color: rgba(245,240,232,0.7);">Academic Insight Score</div>
        <div style="font-size: 1.8rem; color: #f5f0e8; font-weight: 700; font-family: 'Georgia', serif;">
            {silhouette:.3f}
        </div>
        <div style="font-size: 0.7rem; color: rgba(245,240,232,0.5);">Cluster Validation Index</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.info(f"💡 Data includes {len(users_df)} student profiles with synthetic learning records.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Scholar Profile",
        "🔍 Academic Clusters",
        "📚 Course Recommendations",
        "📊 Cohort Analysis"
    ])
    
    with tab1:
        st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0;">👤 Scholar Profile</h2>
        </div>
        """, unsafe_allow_html=True)
        
        user_data = features[features['UserID'] == selected_user].iloc[0]
        cluster = int(user_data['cluster'])
        
        # Scholar identity card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #faf8f5 0%, #f5f0e8 100%);
                    padding: 1.5rem 2rem;
                    border-radius: 16px;
                    border: 1px solid #c4a882;
                    box-shadow: 0 2px 8px 0 rgba(0,0,0,0.05);
                    margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 1.4rem; font-weight: 700; color: #2c1810; font-family: 'Georgia', serif;">
                        {selected_user}
                    </span>
                    <span style="margin-left: 1rem; display: inline-block;">
                        <span style="background: linear-gradient(135deg, #2c1810, #6b3f2a); 
                                     color: #f5f0e8; 
                                     padding: 0.25rem 1rem; 
                                     border-radius: 20px;
                                     font-size: 0.85rem;
                                     font-weight: 600;">
                            📚 Cluster {cluster+1}
                        </span>
                    </span>
                </div>
                <div style="color: #6b5a4a; font-size: 0.9rem;">
                    <span style="background: rgba(196,168,130,0.2); padding: 0.25rem 0.75rem; border-radius: 12px;">
                        🏛️ Validation: {silhouette:.3f}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Demographics
        user_demo = users_df[users_df['UserID'] == selected_user].iloc[0]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a3a5c, #2a5f8f);
                        padding: 1.5rem;
                        border-radius: 16px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 15px 0 rgba(26, 58, 92, 0.2);
                        border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.8rem; opacity: 0.8; font-family: 'Georgia', serif;">🎂 Age</div>
                <div style="font-size: 2.5rem; font-weight: 700; margin: 0.25rem 0;">{user_demo['Age']}</div>
                <div style="font-size: 0.7rem; opacity: 0.6;">years</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #5c1a3a, #8f2a5f);
                        padding: 1.5rem;
                        border-radius: 16px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 15px 0 rgba(92, 26, 58, 0.2);
                        border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.8rem; opacity: 0.8; font-family: 'Georgia', serif;">⚧️ Gender</div>
                <div style="font-size: 2.5rem; font-weight: 700; margin: 0.25rem 0;">{user_demo['Gender']}</div>
                <div style="font-size: 0.7rem; opacity: 0.6;">identity</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a5c3a, #2a8f5f);
                        padding: 1.5rem;
                        border-radius: 16px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 15px 0 rgba(26, 92, 58, 0.2);
                        border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.8rem; opacity: 0.8; font-family: 'Georgia', serif;">📝 Enrollments</div>
                <div style="font-size: 2.5rem; font-weight: 700; margin: 0.25rem 0;">{int(user_data['total_courses'])}</div>
                <div style="font-size: 0.7rem; opacity: 0.6;">courses taken</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Learning profile radar
        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <h3 style="margin: 0;">📊 Learning Profile Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        feature_names = ['total_courses', 'diversity_score', 'avg_rating_enrolled', 'depth_index', 'avg_spent_per_course']
        display_names = ['Engagement', 'Diversity', 'Quality', 'Depth', 'Investment']
        
        radar_df = features[['UserID'] + feature_names].copy()
        for col in feature_names:
            min_val = radar_df[col].min()
            max_val = radar_df[col].max()
            if max_val > min_val:
                radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
            else:
                radar_df[col] = 0
        
        user_radar = radar_df[radar_df['UserID'] == selected_user][feature_names].iloc[0].to_dict()
        cluster_radar = radar_df[features['cluster'] == cluster][feature_names].mean().to_dict()
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[user_radar.get(f, 0) for f in feature_names],
            theta=display_names,
            fill='toself',
            name='Your Profile',
            line=dict(color='#6b3f2a', width=3),
            marker=dict(color='#6b3f2a', size=8),
            fillcolor='rgba(107, 63, 42, 0.2)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[cluster_radar.get(f, 0) for f in feature_names],
            theta=display_names,
            fill='toself',
            name=f'Cohort {cluster+1} Average',
            line=dict(color='#c4a882', width=3, dash='dash'),
            marker=dict(color='#c4a882', size=8),
            fillcolor='rgba(196, 168, 130, 0.15)'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='rgba(196,168,130,0.2)',
                    linecolor='rgba(196,168,130,0.3)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(196,168,130,0.2)',
                    linecolor='rgba(196,168,130,0.3)',
                    tickfont=dict(family='Georgia, serif', size=12, color='#2c1810')
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family='Georgia, serif')
            ),
            margin=dict(l=80, r=80, t=30, b=80),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional learner stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background: #faf8f5; padding: 1rem; border-radius: 12px; border: 1px solid #e8dcc8;">
                <div style="font-size: 0.85rem; color: #6b5a4a;">🏷️ Preferred Subject</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #2c1810; font-family: 'Georgia', serif;">
                    {user_data['preferred_category']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: #faf8f5; padding: 1rem; border-radius: 12px; border: 1px solid #e8dcc8;">
                <div style="font-size: 0.85rem; color: #6b5a4a;">📈 Preferred Level</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #2c1810; font-family: 'Georgia', serif;">
                    {user_data['preferred_level']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0;">🔍 Academic Cluster Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # PCA visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled)
        features['pca1'] = pca_result[:, 0]
        features['pca2'] = pca_result[:, 1]
        
        cluster_colors = ['#2c1810', '#6b3f2a', '#8b5a3c', '#b8956a', '#c4a882', '#d4c4a8', '#e8dcc8', '#f5f0e8']
        
        fig = px.scatter(
            features,
            x='pca1',
            y='pca2',
            color='cluster',
            color_discrete_sequence=cluster_colors[:n_clusters],
            hover_data=['UserID', 'total_courses', 'diversity_score', 'Age', 'Gender'],
            title="Learning Landscape Map",
            labels={'cluster': 'Cohort'}
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6, line=dict(width=1, color='white')))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family='Georgia, serif')
            ),
            title_font=dict(family='Georgia, serif', size=16, color='#2c1810')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <h3 style="margin: 0;">📊 Cohort Characteristics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_summary = features.groupby('cluster').agg({
            'total_courses': 'mean',
            'diversity_score': 'mean',
            'avg_rating_enrolled': 'mean',
            'depth_index': 'mean',
            'avg_spent_per_course': 'mean'
        }).round(2)
        cluster_summary.index = [f'Cohort {i+1}' for i in cluster_summary.index]
        cluster_summary.columns = ['Avg Courses', 'Diversity', 'Avg Rating', 'Depth', 'Avg Spend']
        st.dataframe(cluster_summary.style.highlight_max(axis=0, color='#e8dcc8'))
        
        # Cluster distribution
        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <h3 style="margin: 0;">📈 Cohort Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_counts = features['cluster'].value_counts().sort_index()
        fig_bar = px.bar(
            x=[f'Cohort {i+1}' for i in cluster_counts.index],
            y=cluster_counts.values,
            labels={'x': 'Cohort', 'y': 'Number of Scholars'},
            title="Scholar Distribution Across Cohorts",
            color=cluster_counts.index,
            color_discrete_sequence=cluster_colors[:n_clusters]
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title_font=dict(family='Georgia, serif', size=16, color='#2c1810')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0;">📚 Personalized Course Recommendations</h2>
        </div>
        """, unsafe_allow_html=True)
        
        recs = get_recommendations(selected_user, users_df, transactions_df, courses_df, clusters, features)
        
        if recs.empty:
            st.info("📖 No new recommendations available. You've explored all popular courses in your cohort!")
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #faf8f5, #f5f0e8);
                        padding: 1rem 1.5rem;
                        border-radius: 12px;
                        border-left: 5px solid #6b3f2a;
                        margin-bottom: 1.5rem;">
                <span style="font-family: 'Georgia', serif; color: #2c1810;">
                    🎯 <strong>{len(recs)}</strong> recommended courses based on your learning cohort's preferences
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display as styled dataframe
            display_recs = recs.copy()
            display_recs['Rating'] = display_recs['CourseRating'].apply(lambda x: f"{x} ⭐")
            display_recs = display_recs.drop('CourseRating', axis=1)
            st.dataframe(display_recs.style.highlight_max(axis=0, color='#e8dcc8'))
            
            # Course cards
            st.markdown("""
            <div class="section-header" style="margin-top: 2rem;">
                <h3 style="margin: 0;">📖 Course Details</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, row in recs.iterrows():
                level_badge = f'badge-{row["CourseLevel"].lower()}'
                st.markdown(f"""
                <div class="course-card" style="border-left-color: {cluster_colors[idx % len(cluster_colors)]};">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <strong style="font-size: 1.1rem; color: #2c1810; font-family: 'Georgia', serif;">
                                {row['CourseID']}
                            </strong>
                            <span style="margin-left: 0.75rem; color: #6b5a4a;">{row['CourseCategory']}</span>
                        </div>
                        <span class="badge {level_badge}">{row['CourseLevel']}</span>
                    </div>
                    <div style="margin-top: 0.75rem; display: flex; gap: 2rem; flex-wrap: wrap; font-size: 0.9rem;">
                        <span>📹 {row['CourseType']}</span>
                        <span>⭐ {row['CourseRating']} / 5.0</span>
                        <span>👥 {row['popularity']} scholars</span>
                        <span>📊 Cohort avg: {row['avg_rating']:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0;">📊 Cohort Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature comparison
        feature_to_compare = st.selectbox(
            "Select learning metric to analyze",
            ['total_courses', 'diversity_score', 'avg_rating_enrolled', 'depth_index', 'avg_spent_per_course'],
            format_func=lambda x: {
                'total_courses': '📝 Total Enrollments',
                'diversity_score': '🎯 Subject Diversity',
                'avg_rating_enrolled': '⭐ Average Rating',
                'depth_index': '📈 Learning Depth',
                'avg_spent_per_course': '💰 Investment per Course'
            }.get(x, x)
        )
        
        fig_box = px.box(
            features,
            x='cluster',
            y=feature_to_compare,
            color='cluster',
            color_discrete_sequence=cluster_colors[:n_clusters],
            title=f"Distribution of {feature_to_compare.replace('_', ' ').title()} by Cohort",
            labels={'cluster': 'Cohort', feature_to_compare: feature_to_compare.replace('_', ' ').title()}
        )
        fig_box.update_xaxes(ticktext=[f'Cohort {i+1}' for i in range(n_clusters)], tickvals=list(range(n_clusters)))
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title_font=dict(family='Georgia, serif', size=16, color='#2c1810')
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Two column insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #faf8f5; 
                        padding: 1rem 1.5rem; 
                        border-radius: 12px;
                        border: 1px solid #e8dcc8;
                        margin-top: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #2c1810; font-family: 'Georgia', serif;">
                    🏷️ Preferred Subjects by Cohort
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            pref_cat = features.groupby('cluster')['preferred_category'].agg(lambda x: x.value_counts().index[0]).reset_index()
            pref_cat.columns = ['cluster', 'most_preferred_subject']
            pref_cat['cluster'] = pref_cat['cluster'].apply(lambda x: f'Cohort {x+1}')
            st.dataframe(pref_cat.style.highlight_max(axis=0, color='#e8dcc8'))
        
        with col2:
            st.markdown("""
            <div style="background: #faf8f5; 
                        padding: 1rem 1.5rem; 
                        border-radius: 12px;
                        border: 1px solid #e8dcc8;
                        margin-top: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #2c1810; font-family: 'Georgia', serif;">
                    📊 Cohort Statistics
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            cohort_stats = features.groupby('cluster').agg({
                'total_courses': ['mean', 'min', 'max'],
                'diversity_score': ['mean', 'min', 'max']
            }).round(2)
            cohort_stats.columns = ['Avg Courses', 'Min Courses', 'Max Courses', 'Avg Diversity', 'Min Diversity', 'Max Diversity']
            cohort_stats.index = [f'Cohort {i+1}' for i in cohort_stats.index]
            st.dataframe(cohort_stats.style.highlight_max(axis=0, color='#e8dcc8'))
        
        # Cohort explorer
        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <h4 style="margin: 0;">👥 Explore Cohort Members</h4>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_selected = st.selectbox(
            "Select a cohort to explore",
            sorted(features['cluster'].unique()),
            format_func=lambda x: f'Cohort {x+1}'
        )
        if cluster_selected is not None:
            cluster_users_df = features[features['cluster'] == cluster_selected][
                ['UserID', 'total_courses', 'diversity_score', 'preferred_category', 'Age', 'Gender']
            ]
            st.write(f"**Cohort {cluster_selected+1}:** {len(cluster_users_df)} scholars")
            st.dataframe(cluster_users_df.head(20))
    
    # Academic footer
    st.markdown("""
    <div class="footer">
        🏛️ EduPro Academy — Where Learning Meets Intelligence<br>
        <span style="font-size: 0.8rem; opacity: 0.7;">
            Scholar Segmentation & Personalized Course Discovery Engine
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()