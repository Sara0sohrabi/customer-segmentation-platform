from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sqlalchemy import create_engine
import urllib.parse

app = Flask(__name__)

# ----------------------------------------
# Database config
# ----------------------------------------
SERVER = r"W-BI-DDW\DW2019_HA_P"
DATABASE = "CommonMart"
DRIVER = "SQL Server"

params = urllib.parse.quote_plus(
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
)

SQLALCHEMY_URL = f"mssql+pyodbc:///?odbc_connect={params}"
engine = create_engine(SQLALCHEMY_URL)


# ----------------------------------------
# Helper: fetch table names
# ----------------------------------------
def fetch_tables():
    query = """
    SELECT TABLE_SCHEMA, TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    return pd.read_sql(query, engine)


# ----------------------------------------
# Helper: field stats
# ----------------------------------------
def get_field_stats(table_name):
    query = f"SELECT TOP 1000 * FROM {table_name}"
    df = pd.read_sql(query, engine)

    stats = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())
        missing_percent = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0
        dtype_name = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            suitable = True
            reason = "ستون عددی است و برای KMeans مناسب است."
        elif unique_count <= 10 and non_null_count > 0:
            suitable = True
            reason = "ستون متنی کم‌کاردینال است و با تبدیل به عدد می‌تواند استفاده شود."
        else:
            suitable = False
            reason = "ستون متنی/پرکاردینال است و برای KMeans مناسب نیست."

        stats.append({
            "column_name": col,
            "data_type": dtype_name,
            "non_null_count": non_null_count,
            "unique_count": unique_count,
            "missing_percent": missing_percent,
            "suitable_for_kmeans": suitable,
            "reason": reason
        })

    return stats


# ----------------------------------------
# Route: root
# ----------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is running"})


# ----------------------------------------
# Route: list tables
# ----------------------------------------
@app.route("/api/tables", methods=["GET"])
def api_tables():
    try:
        df = fetch_tables()
        tables = []

        for _, row in df.iterrows():
            full_name = f"[{row['TABLE_SCHEMA']}].[{row['TABLE_NAME']}]"
            tables.append({
                "schema": row["TABLE_SCHEMA"],
                "table_name": row["TABLE_NAME"],
                "full_name": full_name
            })

        return jsonify(tables)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------
# Route: columns
# ----------------------------------------
@app.route("/api/columns", methods=["GET"])
def api_columns():
    try:
        table = request.args.get("table")
        if not table:
            return jsonify({"error": "table parameter is required"}), 400

        query = f"SELECT TOP 1 * FROM {table}"
        df = pd.read_sql(query, engine)

        columns = []
        for col in df.columns:
            columns.append({
                "column_name": col,
                "data_type": str(df[col].dtype)
            })

        return jsonify(columns)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------
# Route: field stats
# ----------------------------------------
@app.route("/api/field-stats", methods=["GET"])
def api_field_stats():
    try:
        table = request.args.get("table")
        if not table:
            return jsonify({"error": "table parameter is required"}), 400

        stats = get_field_stats(table)
        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------
# Route: algorithms
# ----------------------------------------
@app.route("/api/algorithms", methods=["GET"])
def api_algorithms():
    return jsonify([
        {"name": "KMeans"}
    ])


# ----------------------------------------
# Route: run analysis
# ----------------------------------------
@app.route("/api/run-analysis", methods=["POST"])
def api_run_analysis():
    try:
        data = request.get_json() or {}
        table = data.get("table")
        fields = data.get("fields", [])
        algorithm = data.get("algorithm", "KMeans")
        param = int(data.get("param", 3))

        if not table:
            return jsonify({"error": "table is required"}), 400

        if not fields:
            return jsonify({"error": "no fields selected"}), 400

        if algorithm != "KMeans":
            return jsonify({"error": f"algorithm {algorithm} is not supported"}), 400

        query = f"SELECT TOP 1000 {', '.join(fields)} FROM {table}"
        df = pd.read_sql(query, engine)

        if df.empty:
            return jsonify({"error": "selected data is empty"}), 400

        df = df.dropna(subset=fields, how="all")

        if df.empty:
            return jsonify({"error": "all selected rows are empty after cleaning"}), 400

        processed = df.copy()

        for col in processed.columns:
            if pd.api.types.is_numeric_dtype(processed[col]):
                processed[col] = processed[col].fillna(processed[col].median())
            else:
                processed[col] = processed[col].fillna("Unknown")
                le = LabelEncoder()
                processed[col] = le.fit_transform(processed[col].astype(str))

        scaler = StandardScaler()
        X = scaler.fit_transform(processed[fields])

        model = KMeans(n_clusters=param, random_state=42, n_init=10)
        clusters = model.fit_predict(X)

        result_df = df.copy()
        result_df["cluster"] = clusters

        try:
            result_df.to_sql("Clustering_Output", engine, if_exists="replace", index=False)
            save_msg = "نتیجه در جدول Clustering_Output ذخیره شد."
        except Exception as save_err:
            save_msg = f"ذخیره در دیتابیس ناموفق بود: {save_err}"

        summary = f"KMeans با {param} خوشه روی {len(fields)} فیلد اجرا شد. {save_msg}"

        return jsonify({
            "summary": summary,
            "selected_fields": fields,
            "row_count": int(len(result_df)),
            "result": result_df.head(50).to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
