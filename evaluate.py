import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch

# ── Load Data ──
df = pd.read_csv("data/clean_courses.csv")
df = df.dropna(subset=['clean_description'])

# ── TF-IDF Setup ──
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['clean_description'])

# ── Neural Setup ──
neural = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = neural.encode(df['clean_description'].tolist(), convert_to_tensor=True)

# ── Sample Test Queries ──
test_queries = [
    "Python programming for beginners",
    "machine learning data science",
    "web development HTML CSS JavaScript",
    "database SQL management",
    "artificial intelligence deep learning"
]

# ── Metric Functions ──
def precision_at_k(results, k=5):
    """Top-k results mein se kitne relevant hain"""
    return min(k, len(results)) / k

def recall_at_k(results, total_relevant, k=5):
    """Total relevant courses mein se kitne top-k mein mile"""
    if total_relevant == 0:
        return 0
    retrieved = min(k, len(results))
    return retrieved / total_relevant

def hit_rate(results, k=5):
    """Kya top-k mein koi result mila?"""
    return 1 if len(results) > 0 else 0

# ── Run Evaluation ──
print("=" * 60)
print("MODEL EVALUATION REPORT")
print("=" * 60)

tfidf_precisions, tfidf_recalls, tfidf_hits = [], [], []
neural_precisions, neural_recalls, neural_hits = [], [], []

K = 5

for query in test_queries:
    print(f"\nQuery: '{query}'")

    # TF-IDF Results
    qvec = tfidf.transform([query])
    sim = cosine_similarity(qvec, tfidf_matrix)[0]
    top_idx = sim.argsort()[::-1][:K]
    tfidf_results = [df.iloc[i]['course_name'] for i in top_idx if sim[i] > 0]

    # Neural Results
    qemb = neural.encode(query, convert_to_tensor=True)
    nsim = util.cos_sim(qemb, embeddings)[0]
    top_idx_n = torch.topk(nsim, k=K).indices
    neural_results = [df.iloc[i.item()]['course_name'] for i in top_idx_n]

    # Metrics
    total_relevant = len(tfidf_results)  # proxy

    tp = precision_at_k(tfidf_results, K)
    tr = recall_at_k(tfidf_results, total_relevant, K)
    th = hit_rate(tfidf_results, K)

    np_ = precision_at_k(neural_results, K)
    nr = recall_at_k(neural_results, total_relevant, K)
    nh = hit_rate(neural_results, K)

    tfidf_precisions.append(tp)
    tfidf_recalls.append(tr)
    tfidf_hits.append(th)

    neural_precisions.append(np_)
    neural_recalls.append(nr)
    neural_hits.append(nh)

    print(f"  TF-IDF  → Precision@{K}: {tp:.2f} | Recall@{K}: {tr:.2f} | Hit: {th}")
    print(f"  Neural  → Precision@{K}: {np_:.2f} | Recall@{K}: {nr:.2f} | Hit: {nh}")
    print(f"  TF-IDF top results:  {tfidf_results[:3]}")
    print(f"  Neural  top results: {neural_results[:3]}")

# ── Final Summary ──
print("\n" + "=" * 60)
print("AVERAGE SCORES")
print("=" * 60)
print(f"TF-IDF  → Avg Precision@{K}: {sum(tfidf_precisions)/len(tfidf_precisions):.2f}")
print(f"TF-IDF  → Avg Recall@{K}:    {sum(tfidf_recalls)/len(tfidf_recalls):.2f}")
print(f"TF-IDF  → Avg Hit-Rate:      {sum(tfidf_hits)/len(tfidf_hits):.2f}")
print(f"\nNeural  → Avg Precision@{K}: {sum(neural_precisions)/len(neural_precisions):.2f}")
print(f"Neural  → Avg Recall@{K}:    {sum(neural_recalls)/len(neural_recalls):.2f}")
print(f"Neural  → Avg Hit-Rate:      {sum(neural_hits)/len(neural_hits):.2f}")
print("=" * 60)