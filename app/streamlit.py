import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Streamlit Page Config
# =============================
st.set_page_config(page_title="User Segmentation Dashboard", layout="wide")
st.title("📊 App User Segmentation Dashboard")
st.write("Customer Segmentation using K-Means Clustering")

# =============================
# Load Processed Clustered Data
# =============================
@st.cache_data
def load_data():
    path = "data/processed/clustered_users.csv"
    df = pd.read_csv(path)
    return df

df_final = load_data()

# =============================
# Session State for Page Navigation
# =============================
if "page" not in st.session_state:
    st.session_state.page = 1

# =============================
# Navigation Buttons
# =============================
col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("⬅ Previous") and st.session_state.page > 1:
        st.session_state.page -= 1

with col3:
    if st.button("Next ➡") and st.session_state.page < 3:
        st.session_state.page += 1

# =============================
# Display Current Page
# =============================
page = st.session_state.page
st.markdown(f"### Page {page} of 3")

# =============================
# Page 1: Cluster Profile & Customer Identification
# =============================
if page == 1:
    # Calculate cluster counts and percentages
    cluster_counts = df_final["cluster_label"].value_counts()
    total_users = df_final.shape[0]
    cluster_percentages = (cluster_counts / total_users) * 100

    st.subheader("📊 User Segmentation KPIs")
    cols = st.columns(len(cluster_counts) + 1)  # One column per cluster + total
    for i, (cluster_name, count) in enumerate(cluster_counts.items()):
        cols[i].metric(label=f"{cluster_name} (%)", value=f"{cluster_percentages[cluster_name]:.2f}")
    cols[-1].metric(label="Total Users", value=total_users)

    # Also display counts and percentages as a table
    kpi_df = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Count": cluster_counts.values,
    "Percentage (%)": cluster_percentages.values
    })
    st.dataframe(kpi_df)


    st.header("📊 Cluster Profile (Mean Behavior)")
    numeric_cols = df_final.select_dtypes(include=["int64","float64"]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["cluster", "PCA1", "PCA2"]]
    cluster_profile = df_final.groupby("cluster_label")[numeric_cols].mean()
    st.dataframe(cluster_profile)

    st.header("👤 Customer-Level Identification")
    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df_final["cluster_label"].unique())
    )
    cluster_users = df_final[df_final["cluster_label"] == selected_cluster]
    st.write(f"Users in Cluster '{selected_cluster}': {cluster_users.shape[0]}")
    st.dataframe(cluster_users.head(20))

# =============================
# Page 2: Cluster Distribution
# =============================
elif page == 2:
    st.header("📌 Cluster Distribution")
    cluster_counts = df_final["cluster"].value_counts().sort_index()

    fig, ax = plt.subplots()
    cluster_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Users")
    st.pyplot(fig)

# =============================
# Page 3: PCA Cluster Visualization
# =============================
elif page == 3:
    st.header("📈 PCA Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        data=df_final,
        x="PCA1",
        y="PCA2",
        hue="cluster_label",
        palette="Set1",
        ax=ax
    )
    ax.set_title("PCA of User Segments")
    st.pyplot(fig)

    st.markdown("""
    **Legend for Clusters:**  
    - High Engagement  
    - Moderate Engagement  
    - Low Engagement / At-Risk  
    - Occasional Users
    """)