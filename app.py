import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# تنظیمات صفحه
st.set_page_config(page_title="داشبورد تحلیل بیمه", layout="wide")
st.title("📊 داشبورد تحلیل داده‌های بیمه")

# ۱. دریافت و نمایش وضعیت فیلدها از API
st.subheader("وضعیت فیلدها")
try:
    stats = requests.get('http://localhost:5000/api/field-stats').json()
    df_stats = pd.DataFrame(stats)
    st.table(df_stats)
except:
    st.error("خطا در اتصال به API وضعیت فیلدها")

# ۲. بخش تنظیمات و انتخاب الگوریتم
st.subheader("تنظیمات تحلیل")
col1, col2 = st.columns(2)

with col1:
    algorithms = requests.get('http://localhost:5000/api/algorithms').json()
    algo_name = st.selectbox("الگوریتم را انتخاب کنید:", [a['name'] for a in algorithms])

with col2:
    # مثال برای پارامتر پویا
    param_val = st.slider("پارامتر اصلی (مثلاً K در KMeans):", 1, 10, 3)

# ۳. اجرای تحلیل
if st.button("اجرای تحلیل"):
    payload = {"algorithm": algo_name, "param": param_val}
    response = requests.post('http://localhost:5000/api/run-analysis', json=payload).json()
    
    # نمایش خروجی متنی
    st.subheader("نتیجه تحلیل")
    st.write(response['summary'])
    
    # ۴. رسم نمودار با پایتون
    st.subheader("نمودار نتایج")
    chart_data = pd.DataFrame(response['chartData'])
    fig, ax = plt.subplots()
    ax.bar(chart_data['labels'], chart_data['values'])
    st.pyplot(fig)
