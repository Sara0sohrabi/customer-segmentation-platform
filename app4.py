import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="داشبورد تحلیل بیمه", layout="wide")

st.markdown("""
<style>
/* RTL for the whole app */
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}

/* Right-align headings and text */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    direction: rtl;
    text-align: right;
}

/* Tables and dataframe */
.stDataFrame, .stTable {
    direction: rtl;
    text-align: right;
}

/* Keep parameter section LTR */
.param-ltr, .param-ltr * {
    direction: ltr !important;
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 داشبورد تحلیل داده‌ها")

BACKEND = "http://localhost:5000"

# -----------------------------------------
# دریافت جدول‌ها
# -----------------------------------------
try:
    tables_resp = requests.get(f"{BACKEND}/api/tables")
    tables_resp.raise_for_status()
    tables_data = tables_resp.json()

    table_options = {
        f"{t['schema']}.{t['table_name']}": t["full_name"]
        for t in tables_data
    }

    selected_table_label = st.selectbox(
        "یک جدول انتخاب کنید:",
        list(table_options.keys())
    )
    selected_table = table_options[selected_table_label]

except Exception as e:
    st.error(f"خطا در دریافت لیست جدول‌ها: {e}")
    st.stop()

# -----------------------------------------
# آمار فیلدها
# -----------------------------------------
try:
    stats_resp = requests.get(f"{BACKEND}/api/field-stats", params={"table": selected_table})
    stats_resp.raise_for_status()
    stats_data = stats_resp.json()
    stats_df = pd.DataFrame(stats_data)

    st.subheader("ستون‌های جدول و وضعیت هر فیلد")
    st.dataframe(stats_df.rename(columns={
        "column_name": "Column Name",
        "data_type": "Data Type",
        "non_null_count": "Non-Null Count",
        "unique_count": "Unique Count",
        "missing_percent": "Missing Percent",
        "suitable_for_kmeans": "Suitable for KMeans",
        "reason": "Reason"
    }), use_container_width=True)

    suitable_fields = []
    if not stats_df.empty and "suitable_for_kmeans" in stats_df.columns:
        suitable_fields = [
            row["column_name"]
            for _, row in stats_df.iterrows()
            if row["suitable_for_kmeans"] is True
        ]

except Exception as e:
    st.error(f"خطا در دریافت آمار فیلدها: {e}")
    st.stop()

# -----------------------------------------
# انتخاب الگوریتم
# -----------------------------------------
try:
    algo_resp = requests.get(f"{BACKEND}/api/algorithms")
    algo_resp.raise_for_status()
    algorithms = algo_resp.json()
    algorithm_names = [a["name"] for a in algorithms]
    if not algorithm_names:
        algorithm_names = ["KMeans"]
except Exception:
    algorithm_names = ["KMeans"]

st.subheader("انتخاب الگوریتم")
selected_algorithm = st.selectbox("الگوریتم:", algorithm_names)

# -----------------------------------------
# انتخاب فیلدها
# -----------------------------------------
st.subheader("انتخاب فیلدها برای تحلیل")
selected_fields = st.multiselect(
    "فیلدهای مورد نظر را انتخاب کنید:",
    options=stats_df["column_name"].tolist(),
    default=suitable_fields[:3] if suitable_fields else []
)

# -----------------------------------------
# پارامتر K
# -----------------------------------------
st.subheader("تنظیم پارامتر")
st.markdown('<div class="param-ltr">', unsafe_allow_html=True)

k = st.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------
# اجرای تحلیل
# -----------------------------------------
if st.button("🚀 اجرای تحلیل"):
    if not selected_fields:
        st.warning("لطفاً حداقل یک فیلد انتخاب کنید.")
    else:
        try:
            payload = {
                "table": selected_table,
                "fields": selected_fields,
                "algorithm": selected_algorithm,
                "param": k
            }

            res = requests.post(f"{BACKEND}/api/run-analysis", json=payload)
            res.raise_for_status()
            response = res.json()

            if response.get("error"):
                st.error(response["error"])
            else:
                st.success("تحلیل با موفقیت انجام شد.")

                # -----------------------------------------
                # Analysis Result in Persian
                # -----------------------------------------
                st.subheader("نتیجه تحلیل")

                summary = response.get("summary", "")
                persian_summary = f"""
تحلیل با موفقیت انجام شد.

خلاصه نتیجه:
{summary}

تفسیر:
- داده‌های انتخاب‌شده پردازش شدند.
- خوشه‌بندی با موفقیت انجام شد.
- نتایج در جداول و نمودارهای زیر قابل مشاهده هستند.
"""
                st.markdown(persian_summary)

                # -----------------------------------------
                # Cluster Sizes
                # -----------------------------------------
                if response.get("clusterSizes"):
                    st.subheader("Cluster Sizes")
                    cluster_sizes_df = pd.DataFrame(response["clusterSizes"])
                    cluster_sizes_df = cluster_sizes_df.rename(columns={
                        "cluster": "Cluster",
                        "count": "Count"
                    })
                    st.dataframe(cluster_sizes_df, use_container_width=True)

                    fig, ax = plt.subplots()
                    ax.bar(cluster_sizes_df["Cluster"].astype(str), cluster_sizes_df["Count"])
                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("Count")
                    ax.set_title("Distribution of Records by Cluster")
                    st.pyplot(fig)

                # -----------------------------------------
                # Result Table
                # -----------------------------------------
                if response.get("result"):
                    st.subheader("Result Table")
                    result_df = pd.DataFrame(response["result"])
                    result_df = result_df.rename(columns={
                        "cluster": "Cluster"
                    })
                    st.dataframe(result_df, use_container_width=True)
                else:
                    st.info("No tabular result to display.")

                # -----------------------------------------
                # Cluster Profile
                # -----------------------------------------
                if response.get("clusterProfile"):
                    st.subheader("Cluster Profile")
                    profile_df = pd.DataFrame(response["clusterProfile"])
                    profile_df = profile_df.rename(columns={
                        "cluster": "Cluster"
                    })
                    st.dataframe(profile_df, use_container_width=True)

                # -----------------------------------------
                # PCA
                # -----------------------------------------
                if response.get("pcaData"):
                    st.subheader("PCA 2D Visualization")
                    pca_df = pd.DataFrame(response["pcaData"])

                    fig3, ax3 = plt.subplots()
                    for cl in sorted(pca_df["cluster"].unique()):
                        sub = pca_df[pca_df["cluster"] == cl]
                        ax3.scatter(sub["PCA1"], sub["PCA2"], label=f"Cluster {cl}", alpha=0.7)

                    ax3.set_xlabel("Component 1")
                    ax3.set_ylabel("Component 2")
                    ax3.set_title("2D Cluster Visualization with PCA")
                    ax3.legend()
                    st.pyplot(fig3)

        except Exception as e:
            st.error(f"خطا در اجرای تحلیل: {e}")
