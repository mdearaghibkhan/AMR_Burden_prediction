# 🧬 Human Gut AMR Burden Prediction Web App

A **Streamlit-based web application** for predicting **antimicrobial resistance (AMR) burden** in human gut microbiome samples using **machine-learning models and AMR gene abundance data**.

This project is intended for **research and academic use only** and provides interpretable AMR burden scores, resistance mechanism profiles, and exportable results.

---

## 🌍 Web Application Overview

Antimicrobial resistance in the human gut microbiome is a growing public health concern. This web application enables researchers to estimate AMR burden at the sample level using curated AMR gene features and a trained regression model.

The interface is designed to be **light-themed, user-friendly, and suitable for demonstrations, thesis presentations, and GitHub showcasing**.

---

## ✨ Key Features

- 📤 Upload CSV files containing AMR gene abundance data  
- 🧠 Machine-learning–based AMR burden prediction  
- 🧪 Per-sample AMR risk scoring (Low / Moderate / High)  
- 📊 Visualization of resistance mechanism distribution  
- 📁 Export results in **JSON** and **CSV** formats  
- 🎨 Clean and intuitive Streamlit UI  

---

## 🧠 Model Information

| Component | Description |
|--------|------------|
| Model Type | Huber Regressor |
| Input Features | 50 SHAP-selected AMR genes |
| Preprocessing | StandardScaler |
| Output | Continuous AMR Burden Score |
| Risk Categories | Low / Moderate / High |

⚠️ **Note:** The model estimates overall AMR burden and does **not** predict clinical antibiotic susceptibility or treatment outcomes.

---

## 📂 Input Data Requirements

- **File format:** CSV  
- **Rows:** Samples  
- **Columns:** AMR gene names  
- **Index column:** Sample ID  
- **Values:** Normalized gene abundance values  

### Example Input Format

| Sample_ID | gene_1 | gene_2 | ... | gene_50 |
|----------|--------|--------|-----|---------|
| GSM001 | 0.12 | 0.45 | ... | 0.78 |
| GSM002 | 0.22 | 0.33 | ... | 0.91 |

All 50 required AMR genes must be present in the uploaded file.  
If genes are missing, the application will notify the user and provide a downloadable gene list.

---

## 📊 Output Description

For each sample, the application reports:

- **AMR Burden Score:** Continuous numerical value
- **Risk Category:** Low / Moderate / High
- **Resistance Mechanism Profile:** Proportional contribution of mechanisms such as:
  - β-lactamase
  - Aminoglycoside resistance
  - Efflux pump
  - Macrolide efflux
  - Target modification
  - Non-Specific Resistance
- **Interpretation:** Short biological explanation

### Example JSON Output

```json
{
  "Sample_ID": "GSM1306790",
  "AMR_Risk_Score": 5203599.469,
  "Risk_Category": "High",
  "Resistance_Mechanism_Profile": {
    "Non-Specific Resistance": 0.646,
    "β-lactamase": 0.151,
    "Aminoglycoside resistance": 0.131,
    "Efflux pump": 0.030,
    "Macrolide efflux": 0.025,
    "Target modification": 0.016
  },
  "Interpretation": "Resistance is dominated by non-specific background mechanisms with indirect AMR contribution"
}
```

---

## ▶️ How to Run the App Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/amr-burden-predictor.git
cd amr-burden-predictor
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py
├── huber_amr_model.pkl
├── scaler_top50.pkl
├── top50_shap_genes_annotated.csv
├── README.md
└── requirements.txt
```

---

## ⚠️ Disclaimer

This tool is developed **for research and educational purposes only**.  
Predictions should not be used for clinical decision-making or diagnostic purposes without proper experimental and clinical validation.

---

## 👨‍🔬 Intended Audience

- Bioinformatics researchers
- Microbiome scientists
- Computational biology students
- AMR surveillance and public health researchers

---

## 📜 License

This project is released for **academic and non-commercial research use**.  
Please cite appropriately if used in publications or presentations.
