# Misfire Classification Modeling for Roo 🚀

This repository contains our collaborative project with Planned Parenthood Federation of America to enhance **Roo**, an AI chatbot that provides sexual health education to teenagers. Our team developed an error-detection model to classify Roo's responses and provide actionable insights for improving its performance.

---

## 📌 Project Overview

### 🎯 Objectives
1. Develop a real-time error-identification model to monitor Roo's performance.
2. Analyze trends and identify patterns in Roo's misclassifications.
3. Deliver a public-facing report with insights and recommendations for improvement.

### 📝 Outputs
- **Error-Identification Model**: Classifies responses into:
  - True Positives (TP)
  - True Negatives (TN)
  - False Positives (FP)
  - False Negatives (FN)
- **Insights and Trends Analysis**: Highlights key areas for improvement.
- **Final Report**: Comprehensive documentation of findings and recommendations.

---

## 📊 Data Overview

- **Dataset**:
  - 500 labeled examples of user messages and Roo responses.
  - Labels:
    - **True Positives (TP)**: Accurate responses.
    - **False Positives (FP)**: Inaccurate responses.
    - **True Negatives (TN)**: Accurate non-responses.
    - **False Negatives (FN)**: Inaccurate non-responses.

- **Data Processing**:
  - Cleaned text by removing special characters and punctuation.
  - Retained relevant stop words (e.g., pronouns).
  - Performed lemmatization and tokenization.
  - Used pre-trained embeddings (Word2Vec, BERT) for vectorization.

---

## 🔨 Tools and Technologies

- **Programming Languages**: Python
- **Libraries**: Pandas, Scikit-learn, TensorFlow/Keras
- **Tools**: Jupyter Notebooks, VS Code, GitHub, Asana

### Machine Learning Models:
- Logistic Regression
- Random Forest
- Long Short-Term Memory (LSTM)
- Support Vector Machine (SVM)

---

## 🔍 Model Development

### 🏁 Key Steps
1. **Data Understanding & Preprocessing**:
   - Dropped columns with null values.
   - Converted text to numerical features using TF-IDF and BERT embeddings.

2. **Feature Engineering**:
   - Combined embeddings for prompts and responses.

3. **Model Selection**:
   - Evaluated multiple models:
     - **SVM**: Accuracy = 68%
     - **Random Forest**: Accuracy = 71%
     - **Logistic Regression**: Accuracy = 72%
     - **LSTM**: Accuracy = 74% (highest performance).

4. **Hyperparameter Tuning**:
   - Optimized parameters to improve performance and reduce overfitting.

---

## 🚀 Key Results

- **Best Models**:
  - **LSTM**: Accuracy = 74% (training), 74% (validation)
  - **Logistic Regression**: Accuracy = 72%
- **Limitations**:
  - Small dataset size influenced model performance.
  - Domain knowledge and resource allocation constraints.

---

## 🌟 Future Enhancements

- Incorporate user journey data as features.
- Improve precision and specificity of responses.
- Explore the integration of large language models (LLMs) for context-aware responses.

---

## 🧑‍💻 Meet the Team

- **Noely Guzman** (Stony Brook University)  
  [noely.guzman@stonybrook.edu](mailto:noely.guzman@stonybrook.edu)
- **Krit Ravichander** (University of Pittsburgh)  
  [kritikaravichander@gmail.com](mailto:kritikaravichander@gmail.com)
- **Maame Abena Boateng** (University of Texas at Austin)  
  [abenaaboaateng@gmail.com](mailto:abenaaboaateng@gmail.com)
- **Alexis Evans** (Howard University)  
  [alexis.evans.2027@gmail.com](mailto:alexis.evans.2027@gmail.com)

---

## 📝 Acknowledgments

We extend our gratitude to Planned Parenthood Federation of America for their guidance and support throughout this project.

---

### 🛠️ Repository Structure
- `data/`: Contains the processed dataset and related resources.
- `notebooks/`: Jupyter notebooks for preprocessing, training, and evaluation.
- `reports/`: Final report and analysis insights.
- `README.md`: Project overview and details.

---

Feel free to explore the repository. For inquiries, please contact any of the team members above!
