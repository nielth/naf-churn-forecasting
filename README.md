# Membership Churn Forecasting (Thesis Code)

This repository contains supporting code for my Master's thesis:  
**"Membership Churn Forecasting with Classical and Deep Sequential Models"** (University of Agder, 2025).

The project explores multi-horizon churn prediction using both classical ML models and deep sequence models, applied to long-term, sparse membership data.  
Due to data confidentiality, all input data and visualizations have been removed.

---

## 🔧 Repository Structure

> 🚫 No real data included — structure shown for transparency only.

## 📦 Key Techniques

- **Feature Engineering**: Historical behavior, churn gaps, NPS scores, car value estimation via public listings.
- **Multi-Horizon Labeling**: 1-, 2-, and 3-year churn classes created with strict cut-off windows.
- **Modeling**:
  - Classical: Logistic Regression, Random Forest, LightGBM, CatBoost
  - Deep: LSTM, GRU, Informer, Mamba, Titans (PyTorch)
- **Explainability**: SHAP applied to tree-based models for global + local feature attribution.
- **Evaluation**: ROC AUC, macro-F1, and class-wise recall across horizons.

---

## ⚠️ Disclaimer

This code was developed using **confidential data from Norges Automobil-Forbund (NAF)**.  
Due to strict privacy terms, no raw data or outputs are shared in this repository.  
You are free to explore the code structure and adapt it using your own datasets.

---

## 📄 License

MIT License © 2025 Thomas Nielsen
