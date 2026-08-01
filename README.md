# 🎯 AI-Powered Interview Assistant

An AI-powered interview preparation system that helps job seekers practice technical interviews using Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG).

The application retrieves role-specific interview questions, evaluates user responses, analyzes sentiment, and provides intelligent feedback through an interactive Streamlit interface.

---

## 🚀 Features

- AI-powered interview practice
- Semantic question retrieval using Sentence Transformers & FAISS
- Speech-to-text conversion with Whisper
- Sentiment analysis using VADER
- Interactive Streamlit web application
- Performance evaluation using Precision@K and MRR
- Personalized interview feedback

---

## 🛠️ Tech Stack

### Programming Language
- Python

### NLP & AI
- Sentence Transformers
- FAISS
- Whisper
- VADER
- NLTK

### Frameworks & Libraries
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Torch

---

## 📂 Project Structure

```
AI-Powered-Interview-Assistant/
│
├── app.py
├── AI_Interview_Coach.ipynb
├── interview_questions_dataset.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Badr550/AI-Powered-Interview-Assistant.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Select an interview role.
2. Retrieve relevant interview questions using semantic search.
3. Answer the questions by text or voice.
4. Convert speech to text using Whisper.
5. Evaluate the response using NLP techniques.
6. Analyze sentiment with VADER.
7. Display personalized feedback and performance metrics.

---

## 📊 Evaluation

The retrieval system was evaluated using:

- Precision@K
- Mean Reciprocal Rank (MRR)

These metrics were used to measure the relevance and ranking quality of retrieved interview questions.

---

## 👥 Team

- Badr Ahmed
- Maryam Mohamed
- Mariam Sherif
- Hazem Mahmoud

---

## 🙏 Acknowledgments

This project was developed during the **NLP Summer Training Program** organized by the **National Telecommunication Institute (NTI)**.

Special thanks to our instructors for their guidance and continuous support.

---

## 📄 License

This project was developed for educational purposes as part of the NTI NLP Summer Training Program.
