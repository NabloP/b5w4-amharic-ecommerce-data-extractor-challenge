# B5W4: Amharic E-Commerce Data Extractor Challenge – 10 Academy

## 🗂 Challenge Context
This repository documents the submission for 10 Academy’s **B5W4: Amharic E-Commerce Data Extractor Challenge**.  
The goal is to support EthioMart in becoming Ethiopia’s centralized hub for Telegram-based e-commerce by:

- Extracting key business entities (product, price, location) from unstructured Amharic Telegram messages  
- Fine-tuning transformer models for accurate Amharic NER  
- Scoring vendors based on their activity, reach, and pricing to enable data-driven micro-lending  

The project simulates the role of a fintech data analyst building a structured NLP pipeline for intelligent vendor profiling.

### Key Features
- 🧲 Real-time Telegram scraping of e-commerce messages and metadata  
- ✍️ CoNLL-format labeling of Amharic text with Product, Price, and Location entities  
- 🤖 Transformer-based fine-tuning (XLM-Roberta, mBERT) for NER extraction  
- 📊 Vendor-level analytics and micro-lending scorecards  
- 🔍 Model explainability using SHAP and LIME  

---

## 🔧 Project Setup

### 1. Clone the repository:
git clone https://github.com/NabloP/b5w4-amharic-ecommerce-data-extractor-challenge.git
cd b5w4-amharic-ecommerce-data-extractor-challenge

### 2. Create and activate a virtual environment:
**On Windows (PowerShell):**
python -m venv data-extractor-challenge
data-extractor-challenge\Scripts\Activate.ps1

**On macOS/Linux:**
python3 -m venv data-extractor-challenge
source data-extractor-challenge/bin/activate

### 3. Install dependencies:
pip install -r requirements.txt

---

## 📁 Project Structure

<!-- TREE START -->
b5w4-amharic-ecommerce-data-extractor-challenge/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── labeled/
│   ├── outputs/
│   └── logs/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── labeling/
│   ├── modeling/
│   ├── evaluation/
│   └── analytics/
├── scripts/
│   ├── ingest_data.py
│   ├── label_data.py
│   ├── fine_tune_model.py
│   ├── evaluate_models.py
│   └── generate_scorecards.py
├── notebooks/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
<!-- TREE END -->

---

## ✅ Status
- ☑️ Task 1 complete: Ingested Telegram data from 5+ vendors, preprocessed and normalized Amharic messages  
- ☑️ Task 2 complete: Manually labeled 50+ messages in CoNLL format  
- ⬜ Task 3: Fine-tuning underway using XLM-Roberta and Hugging Face pipeline  
- ⬜ Task 4: Model evaluation and comparison to be completed  
- ⬜ Task 5: SHAP & LIME interpretation pending  
- ⬜ Task 6: Vendor scorecard module in progress  

---

## ✍️ Task 2: Manual Labeling for Amharic NER

In this task, we prepared a supervised training dataset for fine-tuning transformer models on Amharic Named Entity Recognition (NER). The steps included:

- 🧪 Sampling 50 diverse, product-rich messages from the cleaned Telegram corpus
- 📄 Translating and tokenizing each message to support annotation
- 🔖 Manually labeling key entities (`B-BRAND`, `B-PRICE`, `B-SIZE`, `B-LOCATION`, `B-CONTACT`) using CoNLL-2003 format
- 🧹 Using regex and heuristics to consistently annotate entities across vendors and formats
- 📁 Saving the structured labels to `data/labeled/labeled_messages.txt` for use in Task 3 model training

This labeling task sets the foundation for our transformer-based NER pipeline and is designed for scalability across thousands of messages in future iterations.

---

## 📦 Key Capabilities
- ✅ Real-time ingestion from Telegram via Telethon  
- ✅ Amharic-friendly tokenization and text normalization  
- ✅ CoNLL-format labeling and entity alignment  
- ✅ Hugging Face-based model fine-tuning for NER  
- ✅ Scorecard logic to support lending decisions  

---

## 🧠 Design Philosophy
This project emphasizes:

- 📦 Modular folder and code design (reusable Python classes)  
- 🔁 Reproducibility via `requirements.txt` and consistent naming  
- 🧪 Explainability using SHAP and LIME  
- 🔍 Transparency in scoring logic for vendor prioritization  

---

## 🚀 Author
Nabil Mohamed  
10 Academy Bootcamp – B5W4 Cohort  
GitHub: https://github.com/NabloP