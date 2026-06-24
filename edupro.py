import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# ----------------------------
# 1. DATA GENERATION (SYNTHETIC)
# ----------------------------
@st.cache_data
def generate_data():
    # Load real Users from the provided CSV
    users_df = pd.read_csv('Users.csv')  # assumes file is in the same directory

    # Generate Courses
    categories = ['Programming', 'Data Science', 'Business', 'Design', 'Languages', 'Marketing', 'Health']
    levels = ['Beginner', 'Intermediate', 'Advanced']
    types = ['Video', 'Interactive', 'Reading']

    courses = []
    for i in range(1, 61):  # 60 courses
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
    courses_df = pd.DataFrame(courses)

    # Generate Transactions
    # Each user enrolls in 3-15 courses, with some bias toward categories based on age/gender
    transactions = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 1)

    for _, user in users_df.iterrows():
        user_id = user['UserID']
        age = user['Age']
        gender = user['Gender']
        # Determine number of courses (more for older, slightly)
        n_courses = np.random.randint(3, 16) if age < 25 else np.random.randint(5, 20)
        # Bias toward certain categories based on gender/age (just for synthetic variety)
        if gender == 'Male':
            preferred = np.random.choice(['Programming', 'Data Science', 'Business'], p=[0.4, 0.3, 0.3])
        else:
            preferred = np.random.choice(['Design', 'Languages', 'Health', 'Marketing'], p=[0.3, 0.3, 0.2, 0.2])
        # Choose courses
        selected_courses = []
        # Ensure at least 2 from preferred category
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
            # Random date
            days = (end_date - start_date).days
            date = start_date + timedelta(days=np.random.randint(0, days))
            # Amount = course price (we'll join later)
            transactions.append({
                'UserID': user_id,
                'CourseID': cid,
                'TransactionDate': date.strftime('%Y-%m-%d'),
                'Amount': None  # filled below
            })

    transactions_df = pd.DataFrame(transactions)
    # Merge with courses to get price
    transactions_df = transactions_df.merge(courses_df[['CourseID', 'Price']], on='CourseID', how='left')
    transactions_df['Amount'] = transactions_df['Price']
    transactions_df.drop('Price', axis=1, inplace=True)

    return users_df, courses_df, transactions_df

# ----------------------------
# 2. FEATURE ENGINEERING
# ----------------------------
def create_learner_features(users_df, transactions_df, courses_df):
    # Aggregate transactions per user
    trans_agg = transactions_df.groupby('UserID').agg(
        total_courses=('CourseID', 'count'),
        total_spent=('Amount', 'sum')
    ).reset_index()

    # Per category count
    trans_cat = transactions_df.merge(courses_df[['CourseID', 'CourseCategory']], on='CourseID')
    cat_pivot = trans_cat.pivot_table(index='UserID', columns='CourseCategory', values='CourseID', aggfunc='count', fill_value=0)
    # Add category counts to features
    features = users_df.merge(trans_agg, on='UserID', how='left').fillna(0)
    features = features.merge(cat_pivot, on='UserID', how='left').fillna(0)

    # Feature engineering
    # Diversity score: number of unique categories enrolled
    cat_cols = [c for c in features.columns if c in categories]
    features['diversity_score'] = (features[cat_cols] > 0).sum(axis=1)

    # Preferred category: category with max enrollment
    features['preferred_category'] = features[cat_cols].idxmax(axis=1)
    # If all zero, set to None
    features['preferred_category'] = features['preferred_category'].where(features[cat_cols].max(axis=1) > 0, 'None')

    # Preferred level: compute from transactions
    trans_level = transactions_df.merge(courses_df[['CourseID', 'CourseLevel']], on='CourseID')
    level_counts = trans_level.groupby(['UserID', 'CourseLevel']).size().unstack(fill_value=0)
    level_cols = ['Beginner', 'Intermediate', 'Advanced']
    for lv in level_cols:
        if lv not in level_counts.columns:
            level_counts[lv] = 0
    features = features.merge(level_counts, on='UserID', how='left').fillna(0)
    # Preferred level
    features['preferred_level'] = features[level_cols].idxmax(axis=1)
    features['preferred_level'] = features['preferred_level'].where(features[level_cols].max(axis=1) > 0, 'None')

    # Average rating of enrolled courses
    trans_rating = transactions_df.merge(courses_df[['CourseID', 'CourseRating']], on='CourseID')
    avg_rating = trans_rating.groupby('UserID')['CourseRating'].mean().reset_index(name='avg_rating_enrolled')
    features = features.merge(avg_rating, on='UserID', how='left').fillna(0)

    # Learning depth index: ratio of advanced to (beginner+intermediate+1)
    features['depth_index'] = features['Advanced'] / (features['Beginner'] + features['Intermediate'] + 1)

    # Average spending per course
    features['avg_spent_per_course'] = features['total_spent'] / features['total_courses'].replace(0, 1)

    # Enrollment frequency: number of transactions per month? Use total courses as proxy.
    # We'll use total courses as engagement.

    # Drop raw category counts for clustering (keep derived features)
    features_for_clustering = features[['total_courses', 'diversity_score', 'avg_rating_enrolled',
                                        'depth_index', 'avg_spent_per_course']].copy()
    # Also include preferred category and level as encoded later? We'll encode separately.

    # Store full features for later use
    return features, features_for_clustering

# ----------------------------
# 3. CLUSTERING
# ----------------------------
def perform_clustering(features_cluster, n_clusters=4):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    silhouette = silhouette_score(scaled, clusters)
    return clusters, scaled, kmeans, scaler, silhouette

# ----------------------------
# 4. RECOMMENDATION ENGINE
# ----------------------------
def get_recommendations(user_id, users_df, transactions_df, courses_df, clusters, features, n_recommend=5):
    # Get cluster of user
    user_idx = users_df[users_df['UserID'] == user_id].index[0]
    cluster = clusters[user_idx]
    # Get all users in same cluster
    cluster_users = users_df.iloc[np.where(clusters == cluster)[0]]['UserID'].tolist()
    # Get courses popular in this cluster: count and average rating
    cluster_trans = transactions_df[transactions_df['UserID'].isin(cluster_users)]
    course_pop = cluster_trans.groupby('CourseID').agg(
        popularity=('CourseID', 'count'),
        avg_rating=('CourseID', lambda x: courses_df[courses_df['CourseID'].isin(x)]['CourseRating'].mean())
    ).reset_index()
    # Merge with course info
    course_pop = course_pop.merge(courses_df, on='CourseID')
    # Filter out courses already taken by the user
    user_courses = transactions_df[transactions_df['UserID'] == user_id]['CourseID'].tolist()
    course_pop = course_pop[~course_pop['CourseID'].isin(user_courses)]
    # Sort by popularity and rating
    course_pop['score'] = (course_pop['popularity'] * 0.6 + course_pop['avg_rating'] * 0.4)  # heuristic
    recommendations = course_pop.sort_values('score', ascending=False).head(n_recommend)
    return recommendations[['CourseID', 'CourseCategory', 'CourseType', 'CourseLevel', 'CourseRating', 'popularity', 'avg_rating']]

# ----------------------------
# 5. STREAMLIT APP
# ----------------------------
def main():
    st.set_page_config(page_title="EduPro Learner Segmentation", layout="wide")
    st.title("🎓 EduPro Student Segmentation & Recommendation Engine")

    # Load data
    with st.spinner("Generating synthetic data..."):
        users_df, courses_df, transactions_df = generate_data()
        features, features_cluster = create_learner_features(users_df, transactions_df, courses_df)

    # Determine optimal clusters using Elbow and Silhouette (we'll do a simple fixed k=4 for demo)
    # But we can compute for a range and display
    st.sidebar.header("Cluster Settings")
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)

    # Perform clustering
    clusters, scaled, kmeans, scaler, silhouette = perform_clustering(features_cluster, n_clusters)
    features['cluster'] = clusters

    # Sidebar: User selection
    st.sidebar.header("Learner Explorer")
    user_ids = users_df['UserID'].tolist()
    selected_user = st.sidebar.selectbox("Select a learner", user_ids)

    # Main area tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Learner Profile", "🔍 Cluster Insights", "📚 Recommendations", "📈 Segment Comparison"])

    with tab1:
        st.subheader(f"Profile for {selected_user}")
        user_data = features[features['UserID'] == selected_user].iloc[0]
        cluster = int(user_data['cluster'])
        st.write(f"**Cluster:** {cluster+1} (out of {n_clusters})")
        st.write(f"**Silhouette Score for this k:** {silhouette:.3f}")

        # Display demographic info
        user_demo = users_df[users_df['UserID'] == selected_user].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Age", user_demo['Age'])
        col2.metric("Gender", user_demo['Gender'])
        col3.metric("Total Courses Enrolled", int(user_data['total_courses']))

        # Feature radar chart
        st.subheader("Learner Behavior Profile (Radar)")
        feature_names = ['total_courses', 'diversity_score', 'avg_rating_enrolled', 'depth_index', 'avg_spent_per_course']
        # Normalize features for radar (min-max)
        radar_df = features[feature_names].copy()
        radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())
        user_radar = radar_df[radar_df['UserID'] == selected_user].iloc[0].to_dict()
        # Also add cluster average
        cluster_radar = radar_df[features['cluster'] == cluster].mean().to_dict()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[user_radar[f] for f in feature_names],
            theta=feature_names,
            fill='toself',
            name='Selected User'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[cluster_radar[f] for f in feature_names],
            theta=feature_names,
            fill='toself',
            name=f'Cluster {cluster+1} Average'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Cluster Insights")
        # PCA visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled)
        features['pca1'] = pca_result[:, 0]
        features['pca2'] = pca_result[:, 1]

        fig = px.scatter(features, x='pca1', y='pca2', color='cluster', 
                         hover_data=['UserID', 'total_courses', 'diversity_score'],
                         title="PCA of Learner Features (colored by cluster)")
        st.plotly_chart(fig, use_container_width=True)

        # Cluster characteristics table
        st.subheader("Cluster Characteristics")
        cluster_summary = features.groupby('cluster').agg({
            'total_courses': 'mean',
            'diversity_score': 'mean',
            'avg_rating_enrolled': 'mean',
            'depth_index': 'mean',
            'avg_spent_per_course': 'mean'
        }).round(2)
        st.dataframe(cluster_summary.style.highlight_max(axis=0))

        # Distribution of clusters
        st.subheader("Cluster Size")
        cluster_counts = features['cluster'].value_counts().sort_index()
        fig_bar = px.bar(x=cluster_counts.index+1, y=cluster_counts.values, labels={'x':'Cluster', 'y':'Number of Learners'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Personalized Course Recommendations")
        # Get recommendations for selected user
        recs = get_recommendations(selected_user, users_df, transactions_df, courses_df, clusters, features)
        if recs.empty:
            st.info("No recommendations available (user might have enrolled in all popular courses).")
        else:
            st.write(f"Top {len(recs)} courses for you (based on cluster popularity and rating):")
            st.dataframe(recs)
            # Display as cards
            for _, row in recs.iterrows():
                with st.expander(f"{row['CourseID']} - {row['CourseCategory']} ({row['CourseLevel']})"):
                    st.write(f"**Type:** {row['CourseType']}")
                    st.write(f"**Rating:** {row['CourseRating']} ⭐")
                    st.write(f"**Popularity in your cluster:** {row['popularity']} enrollments")
                    st.write(f"**Avg rating in cluster:** {row['avg_rating']:.2f}")

    with tab4:
        st.subheader("Segment Comparison")
        # Compare feature distributions across clusters
        feature_to_compare = st.selectbox("Select a feature to compare across clusters", 
                                          ['total_courses', 'diversity_score', 'avg_rating_enrolled', 'depth_index', 'avg_spent_per_course'])
        fig_box = px.box(features, x='cluster', y=feature_to_compare, color='cluster',
                         title=f"Distribution of {feature_to_compare} by Cluster")
        st.plotly_chart(fig_box, use_container_width=True)

        # Cluster profile: preferred categories
        st.subheader("Preferred Course Category by Cluster")
        # Compute most common preferred category per cluster
        pref_cat = features.groupby('cluster')['preferred_category'].agg(lambda x: x.value_counts().index[0]).reset_index()
        pref_cat.columns = ['cluster', 'most_preferred_category']
        st.dataframe(pref_cat)

        # Show all learners in a cluster
        cluster_selected = st.selectbox("Select a cluster to view its learners", sorted(features['cluster'].unique())+1)
        if cluster_selected:
            cluster_users = features[features['cluster']==cluster_selected]['UserID'].tolist()
            st.write(f"Learners in Cluster {cluster_selected}: {len(cluster_users)} users")
            st.dataframe(features[features['cluster']==cluster_selected][['UserID', 'total_courses', 'diversity_score', 'preferred_category']].head(10))

    st.sidebar.markdown("---")
    st.sidebar.write(f"Silhouette Score (k={n_clusters}): {silhouette:.3f}")
    st.sidebar.info("Data is synthetic for demonstration. Real data would yield more meaningful insights.")

if __name__ == "__main__":
    main()