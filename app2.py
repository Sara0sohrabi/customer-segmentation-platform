import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="داشبورد تحلیل بیمه", layout="wide")
st.title("📊 داشبورد تحلیل داده‌ها")

BACKEND = "http://localhost:5000"

# متغیرهای پیش‌فرض
selected_table = None
stats_df = pd.DataFrame()
suitable_fields = []

# -------------------------------------------------
# 1) انتخاب جدول
# -------------------------------------------------
st.subheader("1) انتخاب جدول")
try:
    tables_resp = requests.get(f"{BACKEND}/api/tables")
    tables_resp.raise_for_status()
    tables_data = tables_resp.json()

    table_options = {
        f"{t['schema']}.{t['table_name']}": t["full_name"]
        for t in tables_data
    }

    if table_options:
        selected_table_label = st.selectbox(
            "یک جدول انتخاب کنید:",
            list(table_options.keys())
        )
        selected_table = table_options[selected_table_label]
    else:
        st.warning("هیچ جدولی از بک‌اند دریافت نشد.")

except Exception as e:
    st.error(f"خطا در دریافت لیست جدول‌ها: {e}")

# -------------------------------------------------
# 2) نمایش ستون‌ها و وضعیت آن‌ها
# -------------------------------------------------
if selected_table:
    st.subheader("2) ستون‌های جدول و وضعیت هر فیلد")
    try:
        stats_resp = requests.get(
            f"{BACKEND}/api/field-stats",
            params={"table": selected_table}
        )
        stats_resp.raise_for_status()
        stats_data = stats_resp.json()

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

        if not stats_df.empty and "suitable_for_kmeans" in stats_df.columns:
            suitable_fields = [
                row["column_name"]
                for _, row in stats_df.iterrows()
                if row["suitable_for_kmeans"] is True
            ]

    except Exception as e:
        st.error(f"خطا در دریافت آمار فیلدها: {e}")

    # -------------------------------------------------
    # 3) انتخاب الگوریتم
    # -------------------------------------------------
    st.subheader("3) انتخاب الگوریتم")
    try:
        algo_resp = requests.get(f"{BACKEND}/api/algorithms")
        algo_resp.raise_for_status()
        algorithms = algo_resp.json()
        algorithm_names = [a["name"] for a in algorithms]
        if not algorithm_names:
            algorithm_names = ["KMeans"]
    except Exception:
        algorithm_names = ["KMeans"]

    selected_algorithm = st.selectbox("الگوریتم:", algorithm_names)

    # -------------------------------------------------
    # 4) انتخاب فیلدها برای تحلیل
    # -------------------------------------------------
    st.subheader("4) انتخاب فیلدها برای تحلیل")
    selected_fields = st.multiselect(
        "فیلدهای مورد نظر را انتخاب کنید:",
        options=stats_df["column_name"].tolist() if not stats_df.empty else [],
        default=suitable_fields[:3] if suitable_fields else []
    )

    # -------------------------------------------------
    # 5) نمایش اینکه آیا فیلدها مناسب هستند یا نه
    # -------------------------------------------------
    if not stats_df.empty and selected_fields:
        st.subheader("5) بررسی مناسب بودن فیلدهای انتخاب‌شده")
        chosen_stats = stats_df[stats_df["column_name"].isin(selected_fields)]
        st.dataframe(chosen_stats, use_container_width=True)

    # -------------------------------------------------
    # 6) پارامتر الگوریتم
    # -------------------------------------------------
    st.subheader("6) تنظیم پارامتر")
    k = st.slider("تعداد خوشه‌ها (K):", 2, 10, 3)

    # -------------------------------------------------
    # 7) اجرای تحلیل
    # -------------------------------------------------
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

                    st.subheader("نتیجه تحلیل")
                    st.write(response.get("summary", "summary موجود نیست."))

                    if "result" in response and response["result"]:
                        result_df = pd.DataFrame(response["result"])
                        st.dataframe(result_df, use_container_width=True)

                        if "cluster" in result_df.columns:
                            st.subheader("نمودار تعداد رکورد در هر خوشه")
                            cluster_counts = result_df["cluster"].value_counts().sort_index()

                            fig, ax = plt.subplots()
                            ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
                            ax.set_xlabel("Cluster")
                            ax.set_ylabel("Count")
                            ax.set_title("Distribution of Records by Cluster")
                            st.pyplot(fig)
                    else:
                        st.info("خروجی جدولی برای نمایش وجود ندارد.")

            except Exception as e:
                st.error(f"خطا در اجرای تحلیل: {e}")
