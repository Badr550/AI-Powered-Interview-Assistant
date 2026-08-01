"""
app.py — AI Interview Coach (Streamlit version)
=================================================
This file keeps the team's original pipeline logic (preprocessing,
fine-tuning, FAISS retrieval, evaluate_answer, generate_feedback,
transcribe_audio) EXACTLY as written in the notebook — nothing about
the algorithms/scoring/feedback logic was changed.

Only two things had to be adapted, purely for technical reasons:

1. `!pip install ...` lines don't work inside a normal .py file — install
   the packages once from a terminal instead (see requirements.txt).

2. The whole setup (loading the CSV, preprocessing, fine-tuning the
   embedding model, building the FAISS index) is wrapped inside one
   function decorated with `@st.cache_resource`. This is NOT a change
   to the logic — it's required because Streamlit re-runs the entire
   script from top to bottom every time you click a button. Without
   caching, the app would re-run fine-tuning from scratch on every
   click, which would be unusable. `st.cache_resource` just tells
   Streamlit "run this once, then reuse the result".

Also, the Colab-specific microphone recorder (RECORD_JS /
google.colab.output.eval_js) was removed, since it only works inside a
Colab notebook. It's replaced with Streamlit's own `st.audio_input`
widget, which records audio from the browser mic and hands it to your
EXACT SAME `transcribe_audio()` function — that function itself is
untouched.

Run once before starting the app:
    pip install -r requirements.txt

Then:
    streamlit run app.py
"""

import re
import warnings

import faiss
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Interview Coach", page_icon="🎯", layout="centered")

# ------------------------------------------------------------------
# Path to the pre-built artifacts folder (created by train_and_save.py).
# Run `python train_and_save.py` once, commit the artifacts/ folder to
# GitHub, and this app will just load it — no re-training on startup.
# ------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"


# ====================================================================
# CELL 1 — nltk downloads (unchanged)
# ====================================================================
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer


# ====================================================================
# CELL 3 — Text Preprocessing (unchanged)
# ====================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
sia = SentimentIntensityAnalyzer()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    return " ".join(clean_tokens)


def get_stem_set(text):
    clean = preprocess_text(text)
    return set(stemmer.stem(tok) for tok in clean.split())


# ====================================================================
# SETUP: loads the artifacts produced once by train_and_save.py
# (fine-tuned model, dataframe, embeddings, FAISS index). No training
# happens here — this just loads files from disk. Still wrapped in
# @st.cache_resource so it only loads once per app session instead of
# on every button click.
# ====================================================================
@st.cache_resource(show_spinner="Loading model & data...")
def load_engine(artifacts_dir):
    import os

    df = pd.read_pickle(os.path.join(artifacts_dir, "questions_df.pkl"))
    embedding_model = SentenceTransformer(os.path.join(artifacts_dir, "finetuned_interview_embedder"))
    question_embeddings = np.load(os.path.join(artifacts_dir, "question_embeddings.npy"))
    faiss_index = faiss.read_index(os.path.join(artifacts_dir, "faiss_index.index"))

    import whisper

    whisper_model = whisper.load_model("base")

    return df, embedding_model, question_embeddings, faiss_index, whisper_model


df, embedding_model, question_embeddings, faiss_index, whisper_model = load_engine(ARTIFACTS_DIR)


# ====================================================================
# CELL 12 — RAG Retriever Function (unchanged)
# ====================================================================
def retrieve_questions(job_role, difficulty=None, query=None, top_k=1, exclude_ids=None):
    exclude_ids = exclude_ids or []

    candidates = df[df["job_role"] == job_role].copy()
    if difficulty:
        candidates = candidates[candidates["difficulty"] == difficulty]
    if exclude_ids:
        candidates = candidates[~candidates["question_id"].isin(exclude_ids)]

    if candidates.empty:
        return candidates

    if query is None:
        query = f"important interview question for a {job_role}"

    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    candidate_indices = candidates.index.to_numpy()
    candidate_embeddings = question_embeddings[candidate_indices]

    sims = cosine_similarity(query_vec, candidate_embeddings)[0]
    candidates = candidates.assign(retrieval_score=sims)
    candidates = candidates.sort_values("retrieval_score", ascending=False)

    return candidates.head(top_k)


# ====================================================================
# CELL 13 — Answer Evaluation Engine (unchanged)
# ====================================================================
def keyword_semantic_score(keyword, user_answer_raw, user_answer_clean,
                            exact_score=1.0, semantic_score_val=0.7,
                            semantic_threshold=0.40):
    keyword_stems = get_stem_set(keyword)
    answer_stems = get_stem_set(user_answer_raw)

    if keyword_stems and keyword_stems.issubset(answer_stems):
        return exact_score, "exact"

    sentences = sent_tokenize(user_answer_raw) or [user_answer_raw]
    kw_vec = embedding_model.encode([keyword], convert_to_numpy=True)
    sent_vecs = embedding_model.encode(sentences, convert_to_numpy=True)
    faiss.normalize_L2(kw_vec)
    faiss.normalize_L2(sent_vecs)

    sims = cosine_similarity(kw_vec, sent_vecs)[0]
    best_sim = float(np.max(sims))

    if best_sim >= semantic_threshold:
        return semantic_score_val, "semantic"

    return 0.0, "missing"


def evaluate_answer(user_answer, question_row):
    ideal_answer = question_row["ideal_answer"]
    keywords = [k.strip().lower() for k in question_row["keywords"].split("|")]
    user_answer_clean = preprocess_text(user_answer)

    user_vec = embedding_model.encode([user_answer], convert_to_numpy=True)
    ideal_vec = embedding_model.encode([ideal_answer], convert_to_numpy=True)
    faiss.normalize_L2(user_vec)
    faiss.normalize_L2(ideal_vec)
    semantic_score = float(cosine_similarity(user_vec, ideal_vec)[0][0])

    matched_keywords, exact_keywords, missing_keywords = [], [], []
    per_keyword_scores = []

    for kw in keywords:
        score, match_type = keyword_semantic_score(kw, user_answer, user_answer_clean)
        per_keyword_scores.append(score)
        if match_type == "exact":
            matched_keywords.append(kw)
            exact_keywords.append(kw)
        elif match_type == "semantic":
            matched_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    keyword_score = (sum(per_keyword_scores) / len(keywords)) if keywords else 0

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [user_answer_clean, preprocess_text(ideal_answer)]
    )
    tfidf_score = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

    final_score = (0.5 * semantic_score + 0.3 * keyword_score + 0.2 * tfidf_score) * 100
    final_score = round(final_score, 1)

    if final_score >= 75:
        label = "Excellent"
    elif final_score >= 55:
        label = "Good"
    elif final_score >= 35:
        label = "Needs Improvement"
    else:
        label = "Weak"

    return {
        "semantic_score": round(semantic_score * 100, 1),
        "keyword_score": round(keyword_score * 100, 1),
        "tfidf_score": round(tfidf_score * 100, 1),
        "final_score": final_score,
        "label": label,
        "matched_keywords": matched_keywords,
        "exact_keywords": exact_keywords,
        "missing_keywords": missing_keywords,
    }


# ====================================================================
# CELL 14 — Sentiment Analysis (unchanged)
# ====================================================================
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.5:
        tone_label = "Confident / Positive"
    elif compound >= 0.05:
        tone_label = "Mildly Positive"
    elif compound > -0.05:
        tone_label = "Neutral"
    elif compound > -0.5:
        tone_label = "Mildly Hesitant / Negative"
    else:
        tone_label = "Hesitant / Negative"

    return {
        "compound": round(compound, 3),
        "positive": round(scores["pos"], 3),
        "neutral": round(scores["neu"], 3),
        "negative": round(scores["neg"], 3),
        "tone_label": tone_label,
    }


# ====================================================================
# CELL 15 — Feedback Generation (unchanged)
# ====================================================================
def generate_feedback(evaluation_result, question_row, user_answer):
    criteria_list = [c.strip() for c in question_row["evaluation_criteria"].split("|")]
    missing_kw = evaluation_result["missing_keywords"]
    matched_kw = evaluation_result["matched_keywords"]
    exact_kw = evaluation_result["exact_keywords"]
    semantic_only_kw = [kw for kw in matched_kw if kw not in exact_kw]

    score = evaluation_result["final_score"]
    label = evaluation_result["label"]

    feedback_parts = [f"Overall assessment: {label} ({score}/100)."]

    if exact_kw:
        feedback_parts.append(
            f"Strengths: you correctly used precise professional terms such as "
            f"{', '.join(exact_kw[:4])}."
        )

    if semantic_only_kw:
        feedback_parts.append(
            f"Good conceptual understanding: your answer conveyed the idea behind "
            f"{', '.join(semantic_only_kw)}, though you didn't use the exact term. "
            f"Using the precise professional vocabulary next time would strengthen your answer."
        )

    if missing_kw:
        feedback_parts.append(
            f"To improve: your answer didn't cover {', '.join(missing_kw)}, "
            f"which are important aspects of a complete answer to this question."
        )

    feedback_parts.append(f"This question is typically evaluated on: {'; '.join(criteria_list)}.")

    tone_relevant_categories = ["Behavioral", "Communication"]
    if question_row["category"] in tone_relevant_categories:
        sentiment_result = analyze_sentiment(user_answer)
        tone_label = sentiment_result["tone_label"]

        if tone_label in ["Confident / Positive", "Mildly Positive"]:
            feedback_parts.append(
                f"Tone: Your answer came across as {tone_label.lower()}, "
                f"which is a good sign for a behavioral question — interviewers "
                f"value candidates who sound composed and confident."
            )
        elif tone_label == "Neutral":
            feedback_parts.append(
                "Tone: Your answer was fairly neutral in tone. Adding a bit more "
                "conviction in how you describe your actions can make a stronger impression."
            )
        else:
            feedback_parts.append(
                f"Tone: Your answer came across as {tone_label.lower()}. Try rephrasing "
                f"with more confident, action-oriented language (e.g., 'I handled...' "
                f"instead of 'I'm not sure...')."
            )

    if score < 35:
        feedback_parts.append(
            "Tip: Review the core concept from scratch and structure your "
            "answer with a clear definition followed by an example."
        )
    elif score < 55:
        feedback_parts.append(
            "Tip: Your foundation is there, but add more specific details "
            "or a real-world example to strengthen your answer."
        )
    elif score < 75:
        feedback_parts.append("Tip: Solid answer! Adding the missing points above would make it excellent.")
    else:
        feedback_parts.append(
            "Tip: Excellent, well-rounded answer. Keep this level of detail "
            "and structure in your real interview."
        )

    return "\n".join(feedback_parts)


# ====================================================================
# CELL 16 (partial) — transcribe_audio, UNCHANGED.
# (The Colab mic-recording JS from Cell 16 is not included here since
# it's Colab-only; Streamlit's st.audio_input replaces just that part.)
# ====================================================================
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path, language="en")
    return result["text"].strip()


# ====================================================================
# STREAMLIT UI  (replaces the Gradio interface)
# ====================================================================
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
available_roles = df["job_role"].unique().tolist()


def init_state():
    defaults = {
        "job_role": None,
        "used_ids": [],
        "diff_idx": 0,
        "session_results": [],
        "current_question_id": None,
        "question_display": "Click **Start Interview** to begin.",
        "feedback_display": "",
        "interview_started": False,
        "interview_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def _get_question_display(job_role, difficulty_idx, used_ids):
    retrieved = retrieve_questions(
        job_role=job_role,
        difficulty=DIFFICULTY_LEVELS[difficulty_idx],
        top_k=1,
        exclude_ids=used_ids,
    )
    if retrieved.empty:
        return None, "⚠️ No more questions available for this role."

    q_row = retrieved.iloc[0]
    display_text = (
        f"**Question {difficulty_idx + 1}/3 "
        f"({DIFFICULTY_LEVELS[difficulty_idx]})**\n\n{q_row['question']}"
    )
    return q_row, display_text


def start_interview(job_role):
    st.session_state.job_role = job_role
    st.session_state.used_ids = []
    st.session_state.diff_idx = 0
    st.session_state.session_results = []
    st.session_state.interview_started = True
    st.session_state.interview_done = False
    st.session_state.feedback_display = ""

    q_row, question_display = _get_question_display(job_role, 0, [])
    st.session_state.question_display = question_display
    st.session_state.current_question_id = q_row["question_id"] if q_row is not None else None


def submit_answer(text_answer=None, audio_bytes=None):
    if st.session_state.current_question_id is None:
        st.session_state.feedback_display = "⚠️ Please click 'Start Interview' first."
        return

    if audio_bytes is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        user_answer = transcribe_audio(tmp_path)
    elif text_answer and text_answer.strip():
        user_answer = text_answer.strip()
    else:
        st.session_state.feedback_display = "⚠️ Please type an answer or record your voice."
        return

    question_row = df[df["question_id"] == st.session_state.current_question_id].iloc[0]

    eval_result = evaluate_answer(user_answer, question_row)
    feedback = generate_feedback(eval_result, question_row, user_answer)

    st.session_state.used_ids.append(question_row["question_id"])
    st.session_state.session_results.append({
        "question_id": question_row["question_id"],
        "question": question_row["question"],
        "difficulty": DIFFICULTY_LEVELS[st.session_state.diff_idx],
        "category": question_row["category"],
        "user_answer": user_answer,
        "semantic_score": eval_result["semantic_score"],
        "keyword_score": eval_result["keyword_score"],
        "tfidf_score": eval_result["tfidf_score"],
        "final_score": eval_result["final_score"],
        "label": eval_result["label"],
        "feedback": feedback,
    })

    st.session_state.feedback_display = (
        f"### Score: {eval_result['final_score']}/100 ({eval_result['label']})\n\n{feedback}"
    )

    st.session_state.diff_idx += 1

    if st.session_state.diff_idx < 3:
        q_row, next_question_display = _get_question_display(
            st.session_state.job_role, st.session_state.diff_idx, st.session_state.used_ids
        )
        st.session_state.current_question_id = q_row["question_id"] if q_row is not None else None
        st.session_state.question_display = next_question_display
    else:
        st.session_state.question_display = "✅ **Interview complete!** Scroll down for your final report."
        st.session_state.current_question_id = None
        st.session_state.interview_done = True


def build_final_report():
    if not st.session_state.session_results:
        st.info("No session data yet — complete an interview first.")
        return

    report_df = pd.DataFrame(st.session_state.session_results)
    overall_avg = report_df["final_score"].mean()
    avg_semantic = report_df["semantic_score"].mean()
    avg_keyword = report_df["keyword_score"].mean()
    best_q = report_df.loc[report_df["final_score"].idxmax()]
    worst_q = report_df.loc[report_df["final_score"].idxmin()]

    st.markdown(f"## 📊 Final Interview Report — {st.session_state.job_role}")
    st.markdown(f"**Overall Average Score:** {overall_avg:.1f}/100")
    st.markdown(
        f"**Avg. Semantic Score:** {avg_semantic:.1f}/100  |  "
        f"**Avg. Keyword Score:** {avg_keyword:.1f}/100"
    )
    st.markdown(f"🟢 **Strongest:** [{best_q['difficulty']}] {best_q['question']} — {best_q['final_score']}/100")
    st.markdown(f"🔴 **Weakest:** [{worst_q['difficulty']}] {worst_q['question']} — {worst_q['final_score']}/100")

    if overall_avg >= 75:
        st.success("Excellent performance overall! Well-structured and accurate answers.")
    elif overall_avg >= 55:
        st.info("Good performance — more precise terminology would push you to excellent.")
    elif overall_avg >= 35:
        st.warning("Needs improvement — several key concepts were missing.")
    else:
        st.error("Significant work needed — review core concepts for this role.")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [
        "#4CAF50" if s >= 75 else "#2196F3" if s >= 55 else "#FF9800" if s >= 35 else "#F44336"
        for s in report_df["final_score"]
    ]
    bars = ax.bar(report_df["difficulty"], report_df["final_score"], color=colors, width=0.5)
    ax.axhline(y=overall_avg, color="gray", linestyle="--", linewidth=1.2, label=f"Average ({overall_avg:.1f})")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("Interview Performance Summary", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    for bar, score in zip(bars, report_df["final_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{score}", ha="center", fontweight="bold")
    plt.tight_layout()

    st.pyplot(fig)


# --- Layout ---
st.title("🎯 AI Interview Coach")
st.markdown(
    "An NLP & RAG-powered mock interview system. Choose your role, "
    "answer by **text or voice**, and get instant, grounded feedback."
)

col1, col2 = st.columns([3, 1])
with col1:
    job_role = st.selectbox("Select your job role", available_roles, key="role_select")
with col2:
    st.write("")
    st.write("")
    if st.button("🚀 Start Interview", type="primary", use_container_width=True):
        start_interview(job_role)

st.markdown(st.session_state.question_display)

if st.session_state.interview_started and not st.session_state.interview_done:
    tab_text, tab_voice = st.tabs(["✍️ Text Answer", "🎙️ Voice Answer"])

    with tab_text:
        if st.session_state.get("clear_text_input", False):
            st.session_state.text_answer_input = ""
            st.session_state.clear_text_input = False

        text_answer = st.text_area("Type your answer", key="text_answer_input", height=120)
        if st.button("Submit Text Answer"):
            submit_answer(text_answer=text_answer)
            st.session_state.clear_text_input = True
            st.rerun()

    with tab_voice:
        # Requires Streamlit >= 1.36. On an older version, use
        # st.file_uploader(type=["wav", "mp3", "m4a"]) instead.
        audio_value = st.audio_input("Record your answer", key="voice_answer_input")
        if st.button("Submit Voice Answer"):
            if audio_value is not None:
                submit_answer(audio_bytes=audio_value.getvalue())
                st.rerun()
            else:
                st.warning("⚠️ Please record your voice first.")

if st.session_state.feedback_display:
    st.markdown("---")
    st.markdown(st.session_state.feedback_display)

st.markdown("---")
if st.button("📊 Get Final Report", type="secondary"):
    build_final_report()
