# 💊 CMS Open Payments 2018 — Healthcare Financial Transparency Analysis

> **College Mini Project | Data Science & Machine Learning**
> Uncovering financial influence patterns between pharmaceutical companies and healthcare providers using AI & ML.

---

## 📌 Problem Statement

The **CMS Open Payments Program** (Sunshine Act, 2013) mandates pharmaceutical and medical device companies to publicly report all financial payments made to physicians and teaching hospitals. In 2018 alone, **over $9 billion** was transferred across millions of transactions.

**Key Questions this project answers:**
- 🧐 Which physician specialties receive the highest payments?
- 🏭 Which companies dominate payments and for what purpose?
- 📊 Can we predict payment amounts using machine learning?
- 🧩 What natural groupings (clusters) exist in payment behavior?
- 📋 What policy recommendations can we derive from data?

---

## 🚀 Live Demo

**[Click here to view the live Interactive Dashboard!](https://shweta-tech-creator-cms-open-payments-an-p90wz6.streamlit.app/)**

---

## 🧠 Economic & Business Concepts Applied

| Concept | Application |
|---|---|
| **Demand & Supply** | High-demand specialties attract higher payments |
| **Market Concentration** | Few companies dominate payment volumes (Herfindahl index analysis) |
| **Information Asymmetry** | Transparency data bridges the gap between industry and public |
| **Moral Hazard** | Financial ties may influence prescribing behavior |
| **Price Discovery** | Regression models reveal fair market value of physician services |
| **Risk Analysis** | Cluster-based risk profiling of physician-company relationships |

---

## 🤖 AI & ML Techniques Used

| Technique | Purpose |
|---|---|
| **K-Means Clustering (K=4)** | Segment payment relationships into behavioral groups |
| **Linear Regression** | Predict total payment amounts |
| **Elbow Method** | Optimal cluster count selection |
| **Feature Engineering** | Encode categorical variables, log-transform skewed data |
| **Exploratory Data Analysis** | Trend analysis, payment distribution, specialty breakdown |

---

## 📦 Dataset

| Field | Details |
|---|---|
| **Source** | [CMS Open Payments 2018 — Kaggle](https://www.kaggle.com/datasets/davegords/cms-open-payments-2018) |
| **Size** | ~5 million+ records |
| **Key Columns** | Physician name, specialty, state, company, payment amount, nature of payment |
| **Target Variable** | `Total_Amount_of_Payment_USDollars` |
| **Year** | 2018 |

---

## 🗂️ Project Structure

```
cms-open-payments-analysis/
│
├── 📓 notebooks/
│   └── cms_open_payments_analysis.ipynb   # Full Colab notebook (run in Google Colab)
│
├── 📁 src/
│   └── utils.py                           # Helper functions
│
├── 🚀 app.py                              # Streamlit deployment app
├── 📋 requirements.txt                    # Python dependencies
├── 🙈 .gitignore
└── 📖 README.md
```

---

```

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Shweta-Tech-creator/cms-open-payments-analysis.git
cd cms-open-payments-analysis
```

### 2. Set Up Environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Dataset
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/davegords/cms-open-payments-2018)
- Download `OP_DTL_GNRL_PGYR2018_P01212022.csv`
- In the Streamlit app sidebar, click "Local File Path" and paste the path to your downloaded file.

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

### 5. Open Colab Notebook
- Go to [Google Colab](https://colab.research.google.com/)
- Under the Github tab, search for `Shweta-Tech-creator/cms-open-payments-analysis`
- Open the notebook `notebooks/cms_open_payments_analysis.ipynb`

---

## 📊 Key Findings

- 💰 **Top payment nature**: Food & Beverage, Consulting Fees, Travel & Lodging
- 🩺 **Most compensated specialties**: Orthopedic Surgery, Cardiology, Neurology
- 📈 **Number of payments** is the strongest predictor of total payment amount
- 🏭 **Top 10 companies** account for ~60% of total payment volume

---

## 📋 Policy Recommendations

1. **Stricter thresholds** for high-value consulting payments (>$50K)
2. **Specialty-specific disclosure rules** for high-risk fields (Orthopedics, Cardiology)
3. **Cross-referencing** payment data with prescription behavior databases

---

## 👩‍💻 Built With

- Python 3.10+
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit

---

## 📝 License

This project is for academic/educational purposes only.  
Dataset is publicly available via CMS Open Payments Program.
