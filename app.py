# For deploying text corpora in streamlit cloud
"""
import nltk
import os

nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    from textblob import download_corpora
    download_corpora.download_all()
"""

#---------------------------------
# Importing all the libraries
#------------------------------------
import base64
from dotenv import load_dotenv
load_dotenv()


import os
import io
import re
import time
import json
import clip
import torch
import base64
import joblib
import requests
import pdf2image 
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import language_tool_python
from textblob import TextBlob
from detoxify import Detoxify
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, Dict, Any
import google.generativeai as genai
from diffusers import StableDiffusionPipeline 
from nltk.tokenize import sent_tokenize  
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# transformers 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Text Extraction
from bs4 import BeautifulSoup

# Visualisation
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, f1_score, precision_score, recall_score

# Summarization helper
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#Evaluation 
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Configuring Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
perspective_key = os.getenv("PERSPECTIVE_API_KEY")

# ---------------------------- Streamlit Basic Configuration -----------------------------------------------
st.title("Multi-Dimensional Prompt Response Evaluator")

st.set_page_config(page_title="Multi-Dimensional Prompt Evaluator", layout = "wide")
tab1, tab2, tab4, tab5 = st.tabs(["Analysis Dashboard","Prompt Response Evaluator", "Bias and Toxicity Evaluator", "Image Generator"])
# ------------------------------------Analysis Dashboard -------------------------------------------------------

with tab1:

    # ======== Load datasets =============
    prompt_df = pd.read_csv("Reference_csv_files/Prompt_Evalmetrics.csv")
    response_df = pd.read_csv("Reference_csv_files/PE_Evalmetrics.csv")
    bias_df = pd.read_csv("Reference_csv_files/Bias_responses.csv")
    toxic_perspective = pd.read_csv("Reference_csv_files/Toxicity_response_p.csv")
    toxic_detoxify = pd.read_csv("Reference_csv_files/Toxic_response_d.csv")
    toxic_gemini = pd.read_csv("Reference_csv_files/Toxicity_scores_gemini.csv")
    image_df = pd.read_csv("Reference_csv_files/Image_Evalmetrics.csv")
    inception_gemini = pd.read_csv("Reference_csv_files/inception_scores_per_category.csv")
    inception_SD = pd.read_csv("Reference_csv_files/inception_scores_per_category_SD.csv")
    human_eval_image = pd.read_csv("Reference_csv_files/human_eval_images.csv")



    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to:", [
        "Prompt Engineering and Evaluation of LLM Responses",
        "GPT vs Traditional Model",
        "Bias and Toxicity detection",
        "Text-to Image Generation"
    ])

    # ========== Column mapping for PE ===========
    metrics_mapping = {
        "Word Count": ("Prompt_Word_count", "Word_count"),
        "Length": ("Prompt_length", "Response_length"),
        "Polarity": ("Prompt_Polarity", "Sentiment_Polarity"),
        "Subjectivity": ("Prompt_subjectivity", "Sentiment_subjectivity"),
        "Lexical Diversity": ("Prompt_Lexical_Diversity", "Lexical_Diversity"),
        "Grammar Errors": ("Prompt_Grammar_errors", "Grammar_errors"),
        "Grammar Score": ("Prompt_Grammar_score", "Grammar_score"),
        "Clarity": ("Prompt_Clarity_score", "Clarity_score"),
        "Coherence": ("Prompt_Coherence_score", "Coherence_score")
    }

    # ===============Classification Plot - PE ==================
    def plot_f1_per_class(y_true, y_pred, title, categories=None, figsize=None):
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().iloc[:-3][['precision', 'recall', 'f1-score']]

        # If categories are not provided, take them from y_true
        if categories is None:
            categories = sorted(set(y_true))

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax, colormap='Wistia')

        # Apply categories passed as argument
        ax.set_xticklabels(categories, rotation=0, fontsize=14)

        # Y-axis label and tick size
        ax.set_ylabel("Score", fontsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.tick_params(axis='x', labelsize=24)

        # Title
        ax.set_title(title, fontsize=30)

        # Legend font size
        ax.legend(loc='upper center', fontsize=24)

        # Other formatting
        plt.ylim(0, 1)
        plt.grid(axis='y')

        # Display in Streamlit
        st.pyplot(fig)

    # ========== Classification report - Bias =============

    def plot_f1_bias(y_true, y_pred, title, categories=None, figsize=None):
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().iloc[:-3][['precision', 'recall', 'f1-score']]

        # If categories are not provided, take them from y_true
        if categories is None:
            categories = sorted(set(y_true))

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax, colormap='Paired')

        # Apply categories passed as argument
        ax.set_xticklabels(categories, rotation=0, fontsize=10)

        # Y-axis label and tick size
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)

        # Title
        ax.set_title(title, fontsize=14)

        # Legend font size
        ax.legend(loc='upper right', fontsize=10)

        # Other formatting
        plt.ylim(0, 1)
        plt.grid(axis='y')

        # Display in Streamlit
        st.pyplot(fig)

    # =========== Prompt Engineering Section ================


    if section == "Prompt Engineering and Evaluation of LLM Responses":
        st.header("Prompt Engineering and Evaluation of LLM Responses")

        # Dashboard Discription
        st.markdown("""
        We conducted a comprehensive evaluation of **100 prompts** distributed evenly across **10 categories**:  
        """)
        # Two column layout for categories
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            - 🧠 Analytical
            - 📑 Factual
            - ✍️ Creative Writing
            - 💬 Casual
            - 📘 Instructional 
            """)

        with col2:
            st.markdown("""
            - 🌐 Open-Ended
            - 🌱 Personal Growth
            - 🤔 Philosophical
            - 💼 Professional
            - ⚙️ Technical
            """)
        st.markdown("""
                    #### Models Used
    - **Gemini API** (gemini-2.5-flash)

    #### Evaluation Metrics

    The following metrics were used to measure different aspects of the responses:

    - **Length & Word Count**: Measures verbosity and conciseness
    - **Sentiment Polarity & Subjectivity**: Captures tone and emotional bias
    - **Lexical Diversity**: Evaluates vocabulary richness
    - **Grammar Score & Errors**: Assesses grammatical correctness
    - **Clarity Score**: Indicates comprehensibility of the response
    - **Coherence Score**: Checks logical flow between sentences
    - **Relevance Score**: Measures alignment with the original prompt

    #### Goal of the Dashboard

    This dashboard allows users to:

    - Explore how **LLM responses vary across categories**
    - Compare **prompt design effectiveness**
    - Analyze **objective metrics for response quality**
    - Gain insights into **strengths, weaknesses, and biases of LLM outputs**
                    """)
        
        st.markdown("---")
        st.markdown("##### Per-Category Metric Analysis")

        categories = prompt_df["Category"].unique()

        for metric, (prompt_col, response_col) in metrics_mapping.items():
            if prompt_col in prompt_df.columns and response_col in response_df.columns:
                # Compute per-category averages
                prompt_avg = prompt_df.groupby("Category")[prompt_col].mean().reset_index()
                response_avg = response_df.groupby("Category")[response_col].mean().reset_index()

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                # Prompt plot
                sns.barplot(
                    x="Category", y=prompt_col, data=prompt_avg,
                    palette="Blues", ax=axes[0], ci=None
                )
                axes[0].set_title(f"Prompt {metric}")
                axes[0].set_ylabel(metric)
                axes[0].tick_params(axis='x', rotation=30)
                for label in axes[0].get_xticklabels():
                    label.set_ha("right")

                # Response plot
                sns.barplot(
                    x="Category", y=response_col, data=response_avg,
                    palette="Greens", ax=axes[1], ci=None
                )
                axes[1].set_title(f"Response {metric}")
                axes[1].set_ylabel(metric)
                axes[1].tick_params(axis='x', rotation=30)
                for label in axes[1].get_xticklabels():
                    label.set_ha("right")

                plt.tight_layout()
                st.pyplot(fig)

        # Correlation heatmaps
        st.markdown("##### Correlation Heatmaps")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Prompt heatmap
        numeric_prompt = prompt_df.select_dtypes(include='number')
        prompt_corr = numeric_prompt.corr()
        sns.heatmap(prompt_corr, annot=True, fmt=".2f", cmap="GnBu", center=0, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax[0])
        ax[0].set_title("Prompt Metrics Correlation")

        # Response heatmap
        numeric_response = response_df.select_dtypes(include='number')
        response_corr = numeric_response.corr()
        sns.heatmap(response_corr, annot=True, fmt=".2f", cmap="GnBu", center=0, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax[1])
        ax[1].set_title("Response Metrics Correlation")

        # Plotting 
        plt.tight_layout()
        st.pyplot(fig)

    
    # ============= Bias and Toxicity Section ===========
    elif section == "Bias and Toxicity detection":
        st.header("Bias and Toxicity Detection Dashboard")

        # Dashboard Discription
        st.markdown("""
        This dashboard presents the results of **bias and toxicity detection experiments** conducted using **Large Language Models (LLMs)** across multiple social and linguistic categories.

        - 👩 Gender  
        - 🕌 Religion  
        - 🌍 Ethnicity  
        - 💰 Socioeconomic Status  
        - 🏳️‍🌈 Sexual Orientation  
        - ⚖️ Other sensitive attributes  

        #### Models Used

        - **Bias Detection** → Gemini API
        - **Toxicity Detection** → Detoxify, Perspective API and Gemini API

        #### Evaluation Metrics
        This dashboard evaluates model fairness and safety across text inputs using the following metrics:

        - **Bias Misclassification**: Identifies instances where the model incorrectly predicts bias in text.  
        - **Toxicity Misclassification**: Highlights errors in toxicity classification (false positives/negatives).  
        - **Toxicity Score**: Quantifies the degree of harmful or offensive content detected.  
        - **Classification Report**: Provides precision, recall, and F1-scores for bias and toxicity detection.  
        - **Confusion Matrix**: Visualizes correct vs incorrect predictions for deeper error analysis.  

        ####  Goal of the Dashboard")

        This dashboard allows users to:  

        - Analyze **how LLMs handle bias & toxicity** across categories  
        - Compare **LLMs vs traditional approaches** in fairness and accuracy  
        - Understand **trade-offs between sensitivity and over-flagging**  
        - Gain insights into **safe deployment of LLMs in real-world applications**  
        """)
        # st.write("Placeholder for Bias and Toxicity Detection visualizations...")
        st.markdown("---")

        st.markdown("<h4 style='text-align:center;'>Bias Detection Analysis</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1: 
            # ====== Confusion matrix =============
            st.markdown("###### Confusion Matrix for Bias Detection (Gemini)")
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(bias_df['Category'],bias_df['Response'], labels = sorted(set(bias_df['Category'])))
            
            # Seaborn heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels = sorted(set(bias_df['Category'])), yticklabels=sorted(set(bias_df['Category'])))
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.yaxis.label.set_rotation(90)
            ax.tick_params(axis='y', rotation=0)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            # Display in Streamlit
            st.pyplot(fig)
        
        with col2: 
            # ======== Misclassfication ====
            # Prepare data
            data = {
                "Category": ["Neutral","Sexual Orientation", "Socioeconomic","Ethnicity","Religion", "Gender"],
                "No.of.Misclassifications": [0, 0, 0, 5, 7, 2],
                "Total": [20, 10, 10, 20, 20, 20],
                "Misclassfied Into": ["-","-", "-","Neutral","Neutral, Gender","Neutral"]
            }

            df_misclassifications = pd.DataFrame(data)

            # Misclassification Rate (%) column
            df_misclassifications["Misclassification Rate (%)"] = (df_misclassifications["No.of.Misclassifications"] / df_misclassifications["Total"] * 100 ).round(2)

            # Display table in Streamlit
            st.markdown("###### Misclassification Summary for Bias Detection (Gemini)")
            st.dataframe(df_misclassifications)    

        # ======= Clasification report =======
        plot_f1_bias(bias_df['Category'], bias_df['Response'], "Bias Detection Classification Report", figsize=(10,5))

            # ===Accuracy score======
        accuracy_scores = {"Bias Detection": accuracy_score(bias_df['Category'],bias_df['Response'])
        }

        # Prepare data
        models = list(accuracy_scores.keys())
        scores = list(accuracy_scores.values())

        # ===== Create horizontal bar plot =====
        fig, ax = plt.subplots(figsize=(12, 1))
        ax.barh(models, scores, color=['skyblue', 'lightgreen'])
        ax.set_xlim(0, 1)  # Accuracy ranges from 0 to 1
        ax.set_xlabel("Accuracy Score", fontsize=12)
        ax.set_title("Model Accuracy Comparison", fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Annotate bars with scores
        for i, v in enumerate(scores):
            ax.text(v + 0.01, i, f"{v:.2f}", color='black', fontsize=10, va='center')

        # Display in Streamlit
        st.pyplot(fig)

        st.markdown("---")

        # ============= Toxicity Detection =============
        st.markdown("<h4 style='text-align:center;'>Toxicity Detection Analysis</h4>", unsafe_allow_html=True)
        
        col1, col2,col3 = st.columns(3)
        with col1: 
            # ====== Confusion matrix =============
            st.markdown("###### Confusion Matrix (Perspective API)")
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(toxic_perspective['Category'],toxic_perspective['result'], labels = sorted(set(toxic_perspective['Category'])))
            
            # Seaborn heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels = sorted(set(toxic_perspective['Category'])), yticklabels=sorted(set(toxic_perspective['Category'])))
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.yaxis.label.set_rotation(90)
            ax.tick_params(axis='y', rotation=0)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            # Display in Streamlit
            st.pyplot(fig)
        
        with col2: 
            # ====== Confusion matrix =============
            st.markdown("###### Confusion Matrix (Detoxify)")
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(toxic_detoxify['Category'],toxic_detoxify['result'], labels = sorted(set(toxic_detoxify['Category'])))
            
            # Seaborn heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels = sorted(set(toxic_detoxify['Category'])), yticklabels=sorted(set(toxic_detoxify['Category'])))
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.yaxis.label.set_rotation(90)
            ax.tick_params(axis='y', rotation=0)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            # Display in Streamlit
            st.pyplot(fig)
        
        with col3: 
            # ====== Confusion matrix =============
            st.markdown("###### Confusion Matrix (Gemini API)")
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(toxic_gemini['Category'],toxic_gemini['Response'], labels = sorted(set(toxic_gemini['Category'])))
            
            # Seaborn heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels = sorted(set(toxic_gemini['Category'])), yticklabels=sorted(set(toxic_gemini['Category'])))
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.yaxis.label.set_rotation(90)
            ax.tick_params(axis='y', rotation=0)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            # Display in Streamlit
            st.pyplot(fig)

        # ======== Misclassfication ====
        # Prepare data
        data = {
        "Category": ["Toxicity", "Severe Toxicity", "Identity Attack", "Insult", "Profanity", "Threat"],
        "Total": [18, 15, 17, 17, 16, 17],

        # Misclassifications per model
        "No.of Misclassification (Perspective API)": [16, 15, 17, 12, 15, 17],
        "No.of Misclassification (Detoxify)": [12, 15, 17, 17, 16, 17],
        "No.of Misclassification (Gemini API)": [18, 15, 2, 0, 7, 1],

        # Misclassified Into (optional, can be model-specific if needed)
        "(Perspective API) Misclassified Into ": ["Insult, Non-toxic", "Insult, Non-toxic, Toxicity", "Non-toxic, Toxicity", "Non-toxic, Toxicity","Insult, Non-toxic, Toxicity","Non-toxic, Toxicity"],
        "(Detoxify) Misclassified Into ":["Non-Toxic", "Threat, Non-Toxic, Toxicity", "Non-Toxic, Toxic", "Non-Toxic, Toxicity","Non-Toxic, Toxicity","Non-Toxic, Toxicity"],
        "(Gemini API) Misclassified Into ": ["Identity Attack, Insult","Insult, Threat, Toxicity","Insult","-","Insult","Toxicity"]
        
    
    }
        df_misclassifications = pd.DataFrame(data)
        # Misclassification Rate (%) column
        df_misclassifications["Misclassification Rate % (Perspective API)"] = (df_misclassifications["No.of Misclassification (Perspective API)"] / df_misclassifications["Total"] * 100 ).round(2)
        df_misclassifications["Misclassification Rate % (Detoxify)"] = (df_misclassifications["No.of Misclassification (Detoxify)"] / df_misclassifications["Total"] * 100 ).round(2)
        df_misclassifications["Misclassification Rate % (Gemini API)"] = (df_misclassifications["No.of Misclassification (Gemini API)"] / df_misclassifications["Total"] * 100 ).round(2)


        # Display table
        st.markdown("###### Misclassification Summary for Toxicity Detection")
        st.dataframe(df_misclassifications) 

        # ======== Summary Table ===========
        summary_data = []

         # PERSPECTIVE
        acc_persp = accuracy_score(toxic_perspective['Category'], toxic_perspective['result'])
        f1_persp = f1_score(toxic_perspective['Category'], toxic_perspective['result'], average='macro')
        precision_p = precision_score(toxic_perspective['Category'], toxic_perspective['result'],average='macro')
        recall_p = recall_score(toxic_perspective['Category'], toxic_perspective['result'],average='macro')
        avg_score_persp = toxic_perspective['toxicity'].mean()  # Adjust column if different
        misclass_persp = toxic_perspective[toxic_perspective['Category'] != toxic_perspective['result']]['Category'].value_counts().idxmax()

        summary_data.append(['Perspective', acc_persp, f1_persp,precision_p, recall_p, avg_score_persp, misclass_persp])

        # DETOXIFY
        acc_detox = accuracy_score(toxic_detoxify['Category'], toxic_detoxify['result'])
        f1_detox = f1_score(toxic_detoxify['Category'], toxic_detoxify['result'], average='macro')
        precision_d = precision_score(toxic_detoxify['Category'], toxic_detoxify['result'],average='macro')
        recall_d = recall_score(toxic_detoxify['Category'], toxic_detoxify['result'],average='macro')
        avg_score_detox = toxic_detoxify['toxicity'].mean()
        misclass_detox = toxic_detoxify[toxic_detoxify['Category'] != toxic_detoxify['result']]['Category'].value_counts().idxmax()

        summary_data.append(['Detoxify', acc_detox, f1_detox, precision_d, recall_d, avg_score_detox, misclass_detox])

        # GEMINI
        acc_gemini = accuracy_score(toxic_gemini['Category'], toxic_gemini['Response'])
        f1_gemini = f1_score(toxic_gemini['Category'], toxic_gemini['Response'], average='macro')
        precision_g = precision_score(toxic_gemini['Category'], toxic_gemini['Response'],average='macro')
        recall_g = recall_score(toxic_gemini['Category'], toxic_gemini['Response'],average='macro')
        avg_score_gemini = toxic_gemini['Toxicity_score'].mean()
        misclass_gemini = toxic_gemini[toxic_gemini['Category'] != toxic_gemini['Response']]['Category'].value_counts().idxmax()

        summary_data.append(['Gemini', acc_gemini, f1_gemini, precision_g, recall_g, avg_score_gemini, misclass_gemini])

        summary_df = pd.DataFrame(summary_data, columns=['Model', 'Accuracy', 'Macro F1-Score', 'Precision-Score', 'Recall-Score',
                                                         'Avg Toxicity Score', 'Most Misclassified Category'])
        

        # Apply formatting
        styled_df = summary_df.style.format({
            'Accuracy': "{:.2%}",
            'Macro F1-Score': "{:.2%}",
            'Precision-Score': "{:.2%}",
            'Recall-Score': "{:.2%}",
            'Avg Toxicity Score': "{:.2f}"
        })

        # Display table
        st.markdown("###### Model Performance Summary")
        st.dataframe(styled_df, use_container_width=True)


        # Plotly Radar Chart

        # Radar Chart
        mean_scores = pd.DataFrame({
            'Category': ['toxicity', 'severe_toxicity',	'profanity','threat', 'insult', 'identity_attack'],
            'Detoxify': toxic_detoxify.groupby('Category')['toxicity'].mean().values,
            'Perspective': toxic_perspective.groupby('Category')['toxicity'].mean().values,
            'Gemini': toxic_gemini.groupby('Category')['Toxicity_score'].mean().values
        })

        # Colors from your radar chart
        colors = {
            "Detoxify": {"line": "#a6cee3", "fill": "rgba(166, 206, 227, 0.3)"},       # light blue
            "Gemini": {"line": "#fba23d", "fill": "rgba(253, 180, 98, 0.3)"},      # orange/peach
            "Perspective": {"line": "#bf6e43", "fill": "rgba(177, 89, 40, 0.3)"}     # brown/rust
        }

        # Radar Chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(r=mean_scores['Gemini'],theta=mean_scores['Category'],fill='toself',
            name='Gemini',line=dict(color=colors["Gemini"]["line"]),fillcolor=colors["Gemini"]["fill"]))

        fig.add_trace(go.Scatterpolar(r=mean_scores['Detoxify'],theta=mean_scores['Category'],fill='toself',
            name='Detoxify',line=dict(color=colors["Detoxify"]["line"]),fillcolor=colors["Detoxify"]["fill"]))

        fig.add_trace(go.Scatterpolar( r=mean_scores['Perspective'],theta=mean_scores['Category'],fill='toself',
            name='Perspective',line=dict(color=colors["Perspective"]["line"]),fillcolor=colors["Perspective"]["fill"]))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),title="Toxicity Comparison per Tool", showlegend=True)

        st.plotly_chart(fig)



    # ===============Text-to-Image Generation section ==========
    elif section == "Text-to Image Generation":
        st.header("Text-to-Image Generation Dashboard")
        st.markdown("""
        This dashboard presents the results of **image generation experiments** conducted across **100 prompts**, categorized into five major domains:
        - 🌿 **Nature**
        - 🏙️ **Cityscapes & Structures**
        - 👥 **People**   
        - 🛍️ **Commercial Products**
        - 🎨 **Artistic**

        #### Models Used
        - **Gemini API** → gemini-2.0-flash-preview-image-generation
        - **Stable Diffusion** → stable-diffusion-v1-5

        Each prompt was used to generate images from both models, enabling a **side-by-side evaluation** of their performance.

        #### Evaluation Metrics
        To assess the quality and relevance of generated images, both **automated metrics** and **human evaluations** were employed:

        - **CLIP Score**: Measures image–prompt semantic alignment.  
        - **BLIP Score**: Captures how well the generated image aligns with textual descriptions.  
        - **Inception Score (IS)**: Evaluates image diversity and quality.  
        - **Human Evaluation**: Provides subjective judgments on **realism, creativity, and prompt relevance**.

        #### Goal of the Dashboard
        This dashboard allows users to:
        - Explore **model performance by category**  
        - Compare **Gemini vs Stable Diffusion outputs**  
        - Analyze **objective scores alongside human judgments**  
        - Gain insights into **strengths, limitations, and biases** of different text-to-image generation approaches
        """)
        st.markdown("---")
        st.markdown("#### Computerised Metrics")

        col1, col2 = st.columns(2)


        with col1: 
            # ========= Clip Score ========

            # Melt Gemini & SD clip scores into long format
            df_melted = image_df.melt(id_vars=["Category"], value_vars=["Clip_score_Gemini", "Clip_score_SD"],
                var_name="Score_Type", value_name="Score")

            # Compute average score per category
            avg_scores = df_melted.groupby(["Category", "Score_Type"], as_index=False)["Score"].mean()

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=avg_scores,x="Category",y="Score",hue="Score_Type",palette="Set2",ax=ax)
            ax.set_title("Average CLIP Scores per Category", fontsize=14)
            ax.set_ylabel("Average Score")
            ax.set_xlabel("Category")
            ax.set_ylim(0, 1)
            ax.legend(title="Score Type")

            st.pyplot(fig)

            # Inception score plot
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.errorbar(inception_SD["Category"],inception_SD["Inception_Score_Mean"],
                yerr=inception_SD["Inception_Score_Std"],fmt="o",capsize=5,elinewidth=2,
                marker="s",markersize=8,color="green",ecolor="gray",label="Mean ± Std")

            ax.set_title("Inception Score per Category(Gemini)", fontsize=14)
            ax.set_ylabel("Inception Score")
            ax.set_xlabel("Category")
            ax.set_ylim(1.5, 2.1)
            ax.legend()

            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)


        with col2: 
            # ======= Blip score ==============
            df_melted = image_df.melt(id_vars=["Category"],value_vars=["Blip_score_Gemini", "Blip_score_SD"],
            var_name="Score_Type",value_name="Score")

        # Compute average score per category
            avg_scores = df_melted.groupby(["Category", "Score_Type"], as_index=False)["Score"].mean()

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=avg_scores,x="Category",y="Score",hue="Score_Type",palette="Set2",ax=ax)
            ax.set_title("Average BLIP Scores per Category", fontsize=14)
            ax.set_ylabel("Average Score")
            ax.set_xlabel("Category")
            ax.set_ylim(0, 1)
            ax.legend(title="Score Type")

            st.pyplot(fig)

            # Inception score plot
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.errorbar(inception_SD["Category"],inception_SD["Inception_Score_Mean"],
                yerr=inception_SD["Inception_Score_Std"],fmt="o",capsize=5,elinewidth=2,
                marker="s",markersize=8,color="salmon",ecolor="gray",label="Mean ± Std")

            ax.set_title("Inception Score per Category(Stable Diffusion)", fontsize=14)
            ax.set_ylabel("Inception Score")
            ax.set_xlabel("Category")
            ax.set_ylim(1.5, 2.1)
            ax.legend()

            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig)

        st.markdown("#### Human Evaluated Metrics")

        # Melt the dataframe
        df_melted = human_eval_image.melt(id_vars=["Category"],value_vars=["Score_Gemni","Score_SD"],  # check spelling from st.write
            var_name="Score_Type",value_name="Score")
        
        # plot
        fig_bar, ax_bar = plt.subplots(figsize=(16, 6)) 
        sns.barplot(data=df_melted,x="Category", y="Score", hue="Score_Type",
            palette="Set2", ax=ax_bar, dodge = True,errorbar=None)
        ax_bar.set_title("Average Human Score per Category", fontsize=14)
        ax_bar.set_ylabel("Average Score")
        ax_bar.set_xlabel("Category")
        ax_bar.set_ylim(0, 5)   # updated scale
        ax_bar.legend(title="Score Type")
        plt.xticks(rotation=20)

        st.pyplot(fig_bar)




# ----------------------------- Prompt Response Evaluator -------------------------------------------------

# Load sentence-transformer model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


# =========== Utility Functions for Response Evaluation =================

# Function to get_gemini_response(prompt):
def get_gemini_response(*prompt):
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Function to clean text
def clean_text(text):
    return re.sub(r'[#\*\-]', '', text)

# Function to compute similarity score
def compute_similarity(prompt, response):
    embeddings = similarity_model.encode([prompt, response])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


# Function to compute sentiment
def sentiment_analysis(text):
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3), round(blob.sentiment.subjectivity, 3)

# Lexical diversity
def lexical_diversity(text):
    words = text.split()
    return round(len(set(words)) / len(words), 3) if words else 0.0

# Grammar check
def grammar_score(text):
    tool = language_tool_python.LanguageTool('en-GB')
    matches = tool.check(text)
    error_count = len(matches)
    word_count = len(text.split())
    score = (1-error_count/word_count) * 100 if word_count > 0 else 0.0
    return int(error_count), round(score, 3)

# Clarity Score
def get_clarity(text):
    tool = language_tool_python.LanguageTool('en-GB')
    matches = tool.check(text)
    error_ratio = len(matches) / max(len(text.split()), 1)
    return round(error_ratio, 3)

# Coherence score
def get_coherence(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0.0
    embeddings = similarity_model.encode(sentences, convert_to_tensor=True)
    sims = [util.cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(sentences)-1)]
    avg_sim = sum(sims) / len(sims)
    return round(avg_sim, 3)



# ============ Streamlit UI======================
with tab2:
    st.header("Prompt Response Evaluator")
    st.markdown("Enter a prompt, get a Gemini AI response, evaluate it across multiple metrics and get Feedback on prompt improvement")

    # User Input
    user_prompt = st.text_area("Hello! What Can I help you with today?")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Get Gemini Response
        if st.button("Get Response"):
            if user_prompt.strip():
                with st.spinner("Generating response..."):
                    start_time = time.time()
                    gemini_response = get_gemini_response(user_prompt)
                    cleaned_response = clean_text(gemini_response)
                    end_time = time.time()
                    latency = round(end_time - start_time, 3)

                # Store in session state so buttons can reuse it
                st.session_state['gemini_response'] = gemini_response
                st.session_state['cleaned_response'] = cleaned_response
                st.session_state['Prompt'] = user_prompt


                # Showing Gemini Response
                st.header("Gemini Response:")
                st.write(st.session_state['gemini_response'])

                # show latency
                st.subheader("Response Latency")
                st.write(f"Total response time: **{latency} seconds**")

    with col2:
        # Evaluation Metrics for Prompt and Response
        if st.button("Show Evaluation Metrics"):
            with st.spinner("Evaluating response..."):
                start_time = time.time()
                # ===== Prompt metrics =====
                prompt_text = st.session_state["Prompt"]
                prompt_len = int(len(prompt_text))
                prompt_wc = int(len(prompt_text.split()))
                prompt_polarity, prompt_subjectivity = sentiment_analysis(prompt_text)
                prompt_lex_div = lexical_diversity(prompt_text)
                prompt_grammar_err, prompt_grammar_score = grammar_score(prompt_text)
                prompt_clarity = get_clarity(prompt_text)
                prompt_coherence = get_coherence(prompt_text)

                # ===== Response metrics =====
                resp_text = st.session_state['gemini_response']
                resp_len = int(len(resp_text))
                resp_wc = int(len(resp_text.split()))
                resp_polarity, resp_subjectivity = sentiment_analysis(resp_text)
                resp_relevance = compute_similarity(prompt_text, resp_text)
                resp_lex_div = lexical_diversity(resp_text)
                resp_grammar_err, resp_grammar_score = grammar_score(resp_text)
                resp_clarity = get_clarity(resp_text)
                resp_coherence = get_coherence(resp_text)

                end_time = time.time()
                latency = round(end_time - start_time, 3)

                # ===== Table data =====
                metrics_data = {
                    "Metric": [
                        "Length (chars)",
                        "Word Count",
                        "Sentiment Polarity (Tone)",
                        "Sentiment Subjectivity",
                        "Lexical Diversity",
                        "Grammar Errors",
                        "Grammar Score (%)",
                        "Clarity Score (error %)",
                        "Coherence Score",
                        "Relevance Score"
                    ],
                    "Prompt": [
                        prompt_len,
                        prompt_wc,
                        prompt_polarity,
                        prompt_subjectivity,
                        prompt_lex_div,
                        prompt_grammar_err,
                        prompt_grammar_score,
                        prompt_clarity,
                        prompt_coherence,
                        "—"  # No relevance for prompt 
                    ],
                    "Response": [
                        int(resp_len),
                        int(resp_wc),
                        round(resp_polarity, 3),
                        round(resp_subjectivity, 3),
                        round(resp_lex_div, 3),
                        int(resp_grammar_err),
                        round(resp_grammar_score, 3),
                        round(resp_clarity, 3),
                        round(resp_coherence, 3),
                        round(resp_relevance, 3)
                    ]
                }

                st.subheader("Evaluation Metrics Table")
                pe_df = pd.DataFrame(metrics_data)

                numeric_cols = pe_df.select_dtypes(include=['number']).columns

                st.dataframe(pe_df.style.format("{:.3f}", subset=numeric_cols))

                # Show Latency
                st.subheader("Evaluation Latency")
                st.write(f"Total evaluation time: **{latency} seconds**")

                 # --- Visualization ---
                # ===== Identify count vs score metrics =====
                pe_df = pd.DataFrame(metrics_data)
                pe_df = pe_df.replace("—", 0)
                count_metrics = []
                score_metrics = []

                for metric in pe_df["Metric"]:
                    if "Length" in metric or "Word Count" in metric or "Errors" in metric:
                        count_metrics.append(metric)
                    else:
                        score_metrics.append(metric)
                
                # ===== Create separate DataFrames =====
                count_df = pe_df[pe_df["Metric"].isin(count_metrics)]
                score_df = pe_df[pe_df["Metric"].isin(score_metrics)]

                # ===== Plot Count Metrics =====
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                x = np.arange(len(count_df))
                width = 0.35
                ax1.bar(x - width/2, count_df["Prompt"], width, label="Prompt", color="skyblue")
                ax1.bar(x + width/2, count_df["Response"], width, label="Response", color="lightgreen")
                ax1.set_title("Count Metrics")
                ax1.set_xticks(x)
                ax1.set_xticklabels(count_df["Metric"], rotation=45, ha="right")
                ax1.legend()
                st.pyplot(fig1)

                # ===== Plot Score Metrics (0–1 range) =====
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                x = np.arange(len(score_df))
                ax2.bar(x - width/2, score_df["Prompt"], width, label="Prompt", color="skyblue")
                ax2.bar(x + width/2, score_df["Response"], width, label="Response", color="lightgreen")
                ax2.set_ylim(0, 1)  # Keep scale 0–1
                ax2.set_title("Score Metrics (0–1)")
                ax2.set_xticks(x)
                ax2.set_xticklabels(score_df["Metric"], rotation=45, ha="right")
                ax2.legend()
                st.pyplot(fig2)


# Feedback Section for Prompt Improvement
    with col3:
        if st.button("Feedback on Prompt"):
            with st.spinner("Feedback prompt..."):
                # =========== System Instruction ==============
                prompt_eval = """
                Role: You are an expert in prompt engineering and prompt optimization.
                Objective: Evaluate the given user prompt and provide clear, actionable feedback to improve its structure and clarity so it produces better results from an LLM.
                Evaluation Focus:
                Clarity: Is the intent of the prompt immediately understandable?
                Structure: Is it organized in a logical, step-by-step way?
                Completeness: Does it include necessary context, constraints, and examples for the LLM to follow?
                Specificity: Does it clearly define the desired format, tone, and level of detail?
                Output Format:
                Strengths: Briefly describe what is good about the prompt.
                Areas for Improvement: Identify any vague, missing, or redundant elements.
                Actionable Suggestions: Recommend specific changes to make the prompt more effective (e.g., add constraints, provide examples, specify tone, clarify scope).
                Improved Prompt: Rewrite the prompt with the suggested improvements applied.
                Tone: Professional, constructive, and concise—focus on guiding the user toward a more effective prompt rather than criticizing.
                """
                start_time = time.time()
                response = get_gemini_response(prompt_eval, user_prompt)
                end_time = time.time()
                latency = round(end_time - start_time, 3)
                st.subheader("Prompt Evaluation Feedback:")
                st.write(response)

                # ========== Show Latency ==========
                st.subheader("Feedback Latency")
                st.write(f"Response Time for Feedback: **{latency} seconds**")




# ------------------------------Bias and Toxicity Evaluator --------------------------------------


# --------------Detoxify ------------------
def get_detoxify_scores(text):
    model = Detoxify('original')  # You can also use 'multilingual' if needed
    results = model.predict(text)
    # Convert keys to uppercase and similar style to Perspective API
    formatted_scores = {k.replace("_", " ").title(): v for k, v in results.items()}
    return formatted_scores

# ---------- Perspective API ----------

if not perspective_key:
    st.error("⚠️ Perspective API key not found. Please set PERSPECTIVE_API_KEY in your environment.")
    st.stop()


def get_perspective_scores(text):
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={perspective_key}"

    data = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {}
        }
    }

    response = requests.post(url, data=json.dumps(data))
    result = response.json()

    if "error" in result:
        return {"error": result["error"]["message"]}

    scores = {}
    for attr in result.get('attributeScores', {}):
        scores[attr.replace("_", " ").title()] = result['attributeScores'][attr]['summaryScore']['value']

    return scores
# ----------------------------------
# Formatting Helpers
# --------------------------------------
def format_detoxify_results(detox_res):
    cleaned = {}
    for label, value in detox_res.items():
        try:
            cleaned[label] = round(float(value), 4)
        except Exception:
            cleaned[label] = value
    return cleaned

def format_perspective_results(pers_res):
    if "attributeScores" not in pers_res:
        return pers_res
    cleaned = {}
    for label, scores in pers_res["attributeScores"].items():
        try:
            cleaned[label.lower()] = round(float(scores["summaryScore"]["value"]), 4)
        except Exception:
            cleaned[label.lower()] = scores
    return cleaned

    
# -----------------------------------
# Small Utils
# --------------------------------------

def extract_text_from_url(url:str) -> str:
    # try fetching the URL
    try: 
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        #remove scripts/styles
        for s in soup(["script", "style","noscript"]):
            s.decompose()
        text = ' '. join([p.get_text(separator=' ', strip=True) for p in soup.find_all(['p','h1', 'h2','h3','li'])])
        return text[:30000] # limit size
    except Exception as e:
        return f"[error fetching URL content]{e}"
    
# ---------- Color function ----------
def score_color(score):
    if pd.isna(score):
        return "background-color: #cccccc; color: black;"  # Gray for missing
    if score < 30:
        return "background-color: #2ecc71; color: white;"  # Green
    elif score < 50:
        return "background-color: #f1c40f; color: black;"  # Yellow
    else:
        return "background-color: #e74c3c; color: white;"  # Red

# -----------------------------------
# Streamlit App
# --------------------------------------

with tab4: 
    st.title("Bias and Toxicity Checker")
    col1, col2 = st.columns(2)
    latency_data = []

# ---------------------------------
# Left column - Bias Detection
# ------------------------------------
    with col1:
        st.header("Bias Detection & Feedback")
        bias_prompt = st.text_area("Enter prompt (for Bias Detection):", height=200, key="bias_input")
        
        # Button for Bias Detection
        check_bias = st.button("Check Bias", key="check_bias")

        # Button for Bias Reduction
        reduce_bias = st.button("Reduce Bias", key="reduce_bias")

        # ---- Bias Detection ----
        if check_bias or reduce_bias:
            if not bias_prompt.strip():
                st.warning("Please enter a prompt to analyze.")
            else:
                # Bias Detection Latency
                with st.spinner("Checking for bias..."):
                    start = time.time()
                    Bias_check_prompt = """
    Role: You are a bias detection system that classifies a user’s prompt into exactly one of the following categories:
    Gender, Religion, Ethnicity, Socioeconomic, Sexual Orientation, Neutral.
    Objective:
    Identify if the prompt contains bias.
    If bias is detected, return only one most relevant label from the list above.
    If no bias is detected, label it as Neutral.
    Also, provide an estimated percentage representing the degree of bias in the prompt.
    Output Format:
    If biased: "There is [Category] Bias in the Prompt. It is approximately [X]% biased."
    If neutral: "There is no Bias in the Prompt."
                    """
                    gemini_response = get_gemini_response(bias_prompt, Bias_check_prompt)
                    st.write("**Gemini Response:**", gemini_response)
                    elapsed = round(time.time() - start, 3)
                    latency_data.append({"Model": "Bias Detection (Gemini)", "Latency (seconds)": elapsed})

        # ---- Bias Reduction ----
        if reduce_bias:
            if not bias_prompt.strip():
                st.warning("Enter your prompt first for bias reduction.")
            else:
                with st.spinner("Reducing bias..."):
                    start = time.time()
                    Bias_feedback_prompt = """
    Role: You are a bias reduction system that modifies a user’s prompt to eliminate bias.
    Objective:
    Identify if the prompt contains bias.
    If bias is detected, return a modified version of the prompt that is neutral.
    Output Format:
    If biased: "The modified prompt is: [Modified Prompt]"
    If neutral: "There is no Bias in the Prompt."
                    """
                    gemini_response = get_gemini_response(bias_prompt, Bias_feedback_prompt)
                    st.write("**Gemini Response:**", gemini_response)
                    elapsed = round(time.time() - start, 3)
                    latency_data.append({"Model": "Bias Reduction (Gemini)", "Latency (seconds)": elapsed})

        # ---- Show Latency Table ----
        if latency_data:
            st.markdown("##### Model Latency")
            latency_df = pd.DataFrame(latency_data)
            st.dataframe(latency_df.style.background_gradient(cmap="RdYlGn_r", subset=["Latency (seconds)"]))

    # -----------------------------------
    # Right Column - Toxicity Detection
    # ---------------------------------------
    with col2:
        st.header("Toxicity Detection")
        mode = st.radio("Input type", ["Text", "URL"], horizontal=True)

        if mode == "Text":
            tox_input = st.text_area("Enter text to check toxicity:", height=200, key="tox_text")
        else:
            url = st.text_input("Enter URL to check toxicity:", key="tox_url")
            if url:
                with st.spinner("Fetching URL content..."):
                    tox_input = extract_text_from_url(url)
                    st.text_area("Fetched text (first 3000 chars):", value=tox_input[:3000], height=150, key="fetched_preview")
            else:
                tox_input = ""

        run_tox = st.button("Run Toxicity Checks", key="run_tox")

        if run_tox:
            if not tox_input.strip():
                st.warning("Please enter text or URL to analyze toxicity.")
            else:
                # Detoxify
                with st.spinner("Analyzing..."):
                    start = time.time()
                    detoxify_scores = get_detoxify_scores(tox_input)
                    latency_data.append({"Model": "Toxicity Detection (Detoxify)", "Latency (seconds)": round(time.time() - start, 3)})
                    perspective_scores = get_perspective_scores(tox_input)
                    latency_data.append({"Model": "Toxicity Detection (Perspective API)", "Latency (seconds)": round(time.time() - start, 3)})

                    if "error" in perspective_scores:
                        st.error(f"Perspective API Error: {perspective_scores['error']}")
                    else:
                        # Combine results into a single DataFrame
                        all_attrs = sorted(set(detoxify_scores.keys()) | set(perspective_scores.keys()))
                        df = pd.DataFrame({
                            "Attribute": all_attrs,
                            "Detoxify (%)": [detoxify_scores.get(attr, None) * 100 if detoxify_scores.get(attr) is not None else None for attr in all_attrs],
                            "Perspective API (%)": [perspective_scores.get(attr, None) * 100 if perspective_scores.get(attr) is not None else None for attr in all_attrs]
                        })

                        # Round for cleaner display
                        df["Detoxify (%)"] = df["Detoxify (%)"].round(1)
                        df["Perspective API (%)"] = df["Perspective API (%)"].round(1)

                        # Apply color styling
                        styled_df = df.style.applymap(score_color, subset=["Detoxify (%)", "Perspective API (%)"])

                        st.subheader("📊 Toxicity Detection Comparison")
                        st.write(styled_df)

                # Gemini
                with st.spinner("Checking Gemini..."):
                    start = time.time()
                    toxicity_prompt = f"""
                    You are a toxicity detection system.
                    Analyze the following text for toxic content and provide which kind of Toxicity it is.
                    Text: {tox_input}

                    """
                    gemini_res = get_gemini_response(toxicity_prompt)  # only pass text now
                st.markdown("##### Gemini Results:")
                st.write(gemini_res)
                latency_data.append({"Model": "Toxicity Detection (Gemini)", "Latency (seconds)": round(time.time() - start, 3)})

                # Latency Table
                st.markdown("###### Model Latency")
                latency_df = pd.DataFrame(latency_data)
                st.dataframe(latency_df.style.background_gradient(cmap="RdYlGn_r", subset=["Latency (seconds)"]))
# ----------------------------- Image Generator -------------------------------------------------

with tab5: 
    from diffusers import StableDiffusionPipeline
    import os
    from io import BytesIO
    from google import genai
    from google.genai import types
    from transformers import BlipProcessor, BlipForConditionalGeneration

    from dotenv import load_dotenv
    load_dotenv()
    # -----------------------
    # Page Config
    # -----------------------
    st.set_page_config(page_title="Prompt-to-Image Comparison", layout="wide")
    st.title(" Image Generation: Stable Diffusion vs Gemini API with Evaluation")

    # -----------------------
    # Load Models
    # -----------------------
    @st.cache_resource(show_spinner="Loading Stable Diffusion Pipeline...")
    def load_image_generator():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to(device)
        return pipe

    @st.cache_resource
    def load_clip_model():
        model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        return model, preprocess

    @st.cache_resource
    def load_blip_model():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return processor, model

    pipe = load_image_generator()
    clip_model, clip_preprocess = load_clip_model()
    blip_processor, blip_model = load_blip_model()

    # -----------------------
    # Configure Gemini
    # -----------------------
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_image_gemini(prompt):
        """Generate an image using Gemini 2.0 Flash Preview API (new google-genai SDK)."""
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                st.write(part.text)  # If Gemini returns any text description
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                return image  # Return the generated image

        return None


    # Feedback from Gemini for prompt improvement
    def get_gemini_prompt_feedback(prompt):
    
        prompt_eval = f"""
        Role: You are an expert in prompt engineering for text generation.
                Objective: Review the user’s prompt and provide:
                Short but detailed feedback on how to improve its clarity, structure, and specificity for better LLM output.
                A concise improved version of the prompt that incorporates those improvements.
                Evaluation Focus:
                Is the intent clear and unambiguous?
                Does it specify the desired format, tone, style, and length?
                Is the context complete enough for the model to respond effectively?
                Is it free from unnecessary or vague wording?
                Output Format:
                Feedback: [2–4 sentences of clear, actionable suggestions]
                Improved Prompt: [A short, refined version of the original prompt]

                Tone: Constructive, concise, and easy to follow—focus on what will make the prompt produce the best text generation results.
    """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_eval
        )

        # Return the text output
        return response.text



    # -----------------------
    # Evaluation Functions
    # -----------------------
    def calculate_clip_score(prompt_text, image):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([prompt_text]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            similarity = torch.cosine_similarity(image_features, text_features)
        return similarity.item()

    def blip_caption_similarity(prompt_text, image):
        """Reference-free evaluation: generate caption & measure similarity with prompt."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = blip_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        # Simple similarity: CLIP score between prompt & caption
        text_input = clip.tokenize([prompt_text, caption]).to(device)
        
        text_features = clip_model.encode_text(text_input)
        sim = torch.cosine_similarity(text_features[0].unsqueeze(0), text_features[1].unsqueeze(0))
        return caption, sim.item()

    # -----------------------
    # User Input
    # -----------------------
    prompt = st.text_area("Enter your prompt:", height=100)
    generate_btn = st.button("Generate & Evaluate Images")
    feedback_btn = st.button("Feedback for Prompt Improvement")

    # -----------------------
    # Main Processing
    # -----------------------
    latency_data = []
    if generate_btn and prompt.strip():
        start_time = time.time()
        with st.spinner("Generating image with Stable Diffusion..."):
            sd_image = pipe(prompt).images[0]

        elapsed = round(time.time() - start_time, 3)
        latency_data.append({"Model": "Stable Diffusion Model", "Latency (seconds)": elapsed})

        with st.spinner("Generating image with Gemini API..."):
            start_time = time.time()
            gemini_image = generate_image_gemini(prompt)
            elapsed = round(time.time() - start_time, 3)
            latency_data.append({"Model": "Gemini API Model", "Latency (seconds)": elapsed})

        if gemini_image is None:
            st.error("Gemini API did not return an image.")
        else:
            with st.spinner("Calculating CLIP scores..."):
                sd_clip = calculate_clip_score(prompt, sd_image)
                gemini_clip = calculate_clip_score(prompt, gemini_image)

            with st.spinner("Calculating BLIP relevance scores..."):
                sd_caption, sd_blip_score = blip_caption_similarity(prompt, sd_image)
                gemini_caption, gemini_blip_score = blip_caption_similarity(prompt, gemini_image)

            # Display Images
            col1, col2 = st.columns(2)
            with col1:
                st.image(sd_image, caption=f"Stable Diffusion — {sd_caption}", use_container_width=True)
            with col2:
                st.image(gemini_image, caption=f"Gemini API — {gemini_caption}", use_container_width=True)

            # Display Table
            st.subheader("📊 Evaluation Scores")
            st.table({
                "Model": ["Stable Diffusion", "Gemini API"],
                "CLIP Score": [round(sd_clip, 4), round(gemini_clip, 4)],
                "BLIP Similarity Score": [round(sd_blip_score, 4), round(gemini_blip_score, 4)]
            })

            # ======== Latency Table =======
            if latency_data:
                st.markdown("##### Model Image Generation Latency")
                latency_df = pd.DataFrame(latency_data)
                st.dataframe(latency_df.style.background_gradient(cmap="RdYlGn_r", subset=["Latency (seconds)"]))

    else:
        st.info("Please enter a prompt and click Generate.")


    if feedback_btn and prompt.strip():
        with st.spinner("Evaluating your prompt..."):
            start_time = time.time()
            feedback = get_gemini_prompt_feedback(prompt)
            end_time = time.time()
            st.subheader("Prompt Evaluation Feedback:")
            st.write(feedback)
            latency = round(end_time - start_time,3)
            st.subheader("Feedback Latency")
            st.write(f"Response Time for Feedback: **{latency} seconds**")
            


