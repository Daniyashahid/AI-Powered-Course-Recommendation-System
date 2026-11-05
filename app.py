from datetime import datetime
from flask import Flask, request, jsonify, render_template
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch



app = Flask(__name__)

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
courses_df = pd.read_csv("data/courses.csv")

# -------------------------------
# Step 2: TF-IDF Setup
# -------------------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english' , ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(courses_df['description'])

# -------------------------------
# Step 3: Neural Embeddings Setup
# -------------------------------
neural_model = SentenceTransformer('all-MiniLM-L6-v2')
course_embeddings = neural_model.encode(courses_df['description'].tolist(), convert_to_tensor=True)
# -------------------------------
# Step 4: In-Memory History Storage
# -------------------------------
history = []

# -------------------------------
# Step 5: Flask Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("Index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/recommend")
def recommendation():
    return render_template("recommend.html")

@app.route("/Index")
def index():
    return render_template("Index.html")

# -------------------------------
# API: Recommend Courses
# -------------------------------
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    query = data.get('query', '')
    model_type = data.get('model', 'tfidf')  # default TF-IDF

    results = []

    if model_type == 'tfidf':
        # TF-IDF similarity
        query_vec = tfidf_vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix)
        top_idx = similarity[0].argsort()[::-1][:10]

        for idx in top_idx:
            results.append({
                'course_name': courses_df.iloc[idx]['course_name'],
                'department': courses_df.iloc[idx]['department'],
                'description': courses_df.iloc[idx]['description'],
                'relevance': round(float(similarity[0][idx]) * 100, 2)
            })

    elif model_type == 'neural':
        # Neural embedding similarity
        query_embedding = neural_model.encode(query, convert_to_tensor=True)
        similarity = util.cos_sim(query_embedding, course_embeddings)
        top_idx = torch.topk(similarity[0], k=10).indices

        for idx in top_idx:
            idx = idx.item()
            results.append({
                'course_name': courses_df.iloc[idx]['course_name'],
                'department': courses_df.iloc[idx]['department'],
                'description': courses_df.iloc[idx]['description'],
                'relevance': round(float(similarity[0][idx]) * 100, 2)
            })

        # Save to history
    history.append({
        'query': query,
        'model': model_type,
        'results': results
    })
    # 🔹 SAVE SEARCH TO DATABASE (Dashboard ke liye)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history (query, model, timestamp) VALUES (?, ?, ?)",
        (query, model_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    conn.commit()
    conn.close()

    return jsonify(results)

# -------------------------------
# API: Get Search History
# -------------------------------
@app.route('/api/history', methods=['GET'])
def api_history():
    conn = get_db_connection()
    historys = conn.execute(
        "SELECT query, model, timestamp FROM history ORDER BY id DESC"
    ).fetchall()
    conn.close()

    history_list = []
    for row in historys:
        history_list.append({
            "query": row["query"],
            "model": row["model"],
            "timestamp": row["timestamp"]
        })

    return jsonify(history_list)

# -------------------------------
# API: Clear Search History
# -------------------------------
@app.route('/api/clear_history', methods=['POST'])
def api_clear_history():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()
    conn.close()

    return jsonify({"status": "history cleared"})

# -------------------------------
# API: Save Recommendations
# -------------------------------
@app.route('/api/save', methods=['POST'])
def api_save():
    try:
        data = request.json
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
                INSERT INTO saved_recommendations
                (course_name, department, description, relevance, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
            data["course_name"],
            data["department"],
            data["description"],
            data["relevance"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return jsonify({"status": "saved"}), 200

    except Exception as e:
        print("SAVE ERROR:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------
# GET SAVED RECOMMENDATIONS
# -------------------------------

@app.route('/api/saved', methods=['GET'])
def api_get_saved():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM saved_recommendations ORDER BY id DESC"
    ).fetchall()
    conn.close()

    saved_list = []
    for row in rows:
        saved_list.append({
            "id": row["id"],
            "course_name": row["course_name"],
            "department": row["department"],
            "description": row["description"],
            "relevance": row["relevance"],
            "timestamp": row["timestamp"]
        })

    return jsonify(saved_list)

@app.route('/api/saved/delete/<int:rec_id>', methods=['DELETE'])
def api_delete_saved(rec_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM saved_recommendations WHERE id = ?", (rec_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -------------------------------
# Database Connection Function
# -------------------------------

def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------------
# Model Comparison
# -------------------------------

@app.route('/api/compare', methods=['POST'])
def api_compare():
    data = request.json
    query = data.get('query', '')

    # ---------------- TF-IDF RESULTS ----------------
    tfidf_results = []

    query_vec = tfidf_vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_idx = similarity[0].argsort()[::-1][:5]

    for idx in top_idx:
        tfidf_results.append({
            "course_name": courses_df.iloc[idx]['course_name'],
            "department": courses_df.iloc[idx]['department'],
            "description": courses_df.iloc[idx]['description'],
            "relevance": round(float(similarity[0][idx]) * 100, 2)
        })

    # ---------------- NEUTRAL RESULTS ----------------
    neural_results = []

    query_embedding = neural_model.encode(query, convert_to_tensor=True)
    similarity_neural = util.cos_sim(query_embedding, course_embeddings)
    top_idx_neural = torch.topk(similarity_neural[0], k=5).indices

    for idx in top_idx_neural:
        idx = idx.item()
        neural_results.append({
            "course_name": courses_df.iloc[idx]['course_name'],
            "department": courses_df.iloc[idx]['department'],
            "description": courses_df.iloc[idx]['description'],
            "relevance": round(float(similarity_neural[0][idx]) * 100, 2)
        })

    # ---------------- SAVE SEARCH TO HISTORY ----------------
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (query, model, timestamp) VALUES (?, ?, ?)",
        (query, "compare", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

    # ---------------- FINAL RESPONSE ----------------
    return jsonify({
        "tfidf": tfidf_results,
        "neural": neural_results
    })

# -------------------------------
# Saved Table
# -------------------------------
def create_saved_table():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                model TEXT,
                timestamp TEXT
            )
        """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_name TEXT,
            department TEXT,
            description TEXT,
            relevance REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()
create_saved_table()

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)