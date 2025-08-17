# Multi-Dimensional Evaluation and Application of Generative AI: A Study of Prompt Effectiveness, Ethical Risks, and Real-World Use Cases

This repository contains a collection of experiments and implementations around **Prompt Engineering and Evaluation of LLMs, GPT vs Traditional Models,Bias & Toxicity Detection, Text-to-Image Generation, and Resume Evaluation**.  

The project explores different aspects of **Generative AI** using **Gemini API, GPT models, Transformers, Stable Diffusion, and Traditional NLP methods**, alongside various **evaluation metrics and visualizations**.  

---

## 📑 Table of Contents  
1. [Prompt Engineering & Evaluation of LLM Response](#a-prompt-engineering-and-evaluation-of-llm-response)  
2. [GPT vs Traditional Model Performance](#b-gpt-vs-traditional-model-performance-for-tasks)  
3. [Bias & Toxicity Detection](#c-bias-and-toxicity-detection-in-the-text-using-llm)  
4. [Text-to-Image Generation & Relevance Evaluation](#d-text-to-image-generation-using-llm--relevance-evaluation)  
5. [Resume Evaluation & Feedback](#e-resume-evaluation-and-feedback-using-llm)  
6. [Technologies Used](#technologies-used)  
7. [Visualizations](#visualizations)  
8. [How to Run](#how-to-run)  
9. [Future Work](#future-work)  

---

## A. Prompt Engineering and Evaluation of LLM Response  

- **100 prompts** across 10 categories: *Factual, Creative Writing, Instructional, Philosophical, Casual, Analytical, Professional, Personal Growth, Technical, Open-ended*.  
- Applied to **Gemini model**, responses cleaned and evaluated.  

### 📝 Evaluation Metrics  
- Response Length & Word Count  
- Sentiment Polarity & Subjectivity  
- Relevance to Prompt  
- Lexical Diversity  
- Grammar Errors & Grammar Score  

### 📊 Visualizations  
- Response length by category  
- Relevance scores  
- Sentiment polarity & subjectivity by category  
- Grammar score distribution  
- Lexical diversity by category  
- Correlation heatmap & Pairplot  

✅ Automated pipeline for evaluation  

---

## B. GPT Vs Traditional Model Performance for Tasks  

### 1. Sentiment Analysis  
- **Dataset:** Amazon Polarity  
- **Traditional Approach:** Preprocessing → Vectorization → Logistic Regression/Naïve Bayes → Hyperparameter tuning → Classification Report  
- **Transformer Approach:** HuggingFace pipeline → Training & Evaluation  
- **Result:** Traditional Model outperformed Transformer for this task.  

### 2. Text Summarization  
- **Dataset:** CNN/DailyMail  
- **Traditional:** Sumy summarizer + ROUGE score  
- **Transformer (BART, Pegasus, etc.):** Tokenization → Summary generation  
- **Evaluation Metrics:**  
  - Cosine & Semantic Similarity  
  - Lexical Diversity  
  - Compression Ratio  
  - Sentiment shift (Article vs Summary)  
  - Repetition & Readability Score  

**Visualizations:** Comparative charts of all metrics  

---

## C. Bias and Toxicity Detection in the Text using LLM  

### 1. Bias Detection  
- **100 biased prompts** across categories: Gender, Religion, Ethnicity, Socioeconomic, Sexual Orientation, Neutral  
- LLM predicted type of bias  
- **Evaluation:** Classification Report + Confusion Matrix  

### 2. Toxicity Detection  
- **Models Used:**  
  - Perspective API  
  - Detoxify  
  - Gemini API (custom toxicity scoring)  

- **Evaluation Metrics:** Toxicity, Severe Toxicity, Insult, Profanity, Threat, Identity Attack  
- **Visualizations:**  
  - Confusion matrices & classification reports  
  - Radar chart for overall comparison  

---

## D. Text-to-Image Generation using LLM & Relevance Evaluation  

### 1. Gemini API  
- 100 creative prompts → Generated Images  
- **Evaluation:** CLIP score  

### 2. Stable Diffusion  
- Same prompts applied to SD model  
- **Evaluation:** CLIP score  

**Result:** Gemini outperformed Stable Diffusion in terms of relevance.  

---

## E. Resume Evaluation and Feedback using LLM  

- Extract resume text (PDF parsing)  
- Match percentage & missing keywords  
- Skill matching against job description  
- Detailed metrics for strengths/weaknesses  
- **Personalized feedback** on resume improvement using Gemini API  

---

## 🛠️ Technologies Used  

- **LLMs & APIs:** Gemini API, OpenAI GPT, HuggingFace Transformers  
- **Traditional NLP:** Scikit-learn, Sumy  
- **Evaluation:** ROUGE, Cosine Similarity, CLIP, Readability, Toxicity Metrics  
- **Visualization:** Matplotlib, Seaborn, Radar charts, Pairplots, Heatmaps  
- **Deployment:** Streamlit  

---

## 📊 Visualizations  

- Correlation heatmaps  
- Pairplots for feature relationships  
- Sentiment distributions  
- Bias & Toxicity confusion matrices  
- Radar chart comparisons  
- Resume analysis dashboards  

---

## 🚀 How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/repo-name.git
   cd repo-name

