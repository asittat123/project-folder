# Heart Failure Prediction - Multi-Model Classification Dashboard

BITS Pilani Work Integrated Learning Programmes (WILP)  
M.Tech (AIML) - Machine Learning  
Assignment - 2  

---

## a. Problem Statement
Cardiovascular disease remains the leading cause of global mortality. Early, accurate detection and risk stratification of cardiovascular conditions allow timely preventative care and clinical intervention. The objective of this project is to develop, evaluate, and deploy an end-to-end classification pipeline that predicts whether a patient is likely to have heart disease (`HeartDisease`: 0 = Normal/Absent, 1 = Heart Disease/Present) using biometric and diagnostic observations.

Five core machine learning classifiers are implemented, evaluated, serialized, and deployed via an interactive Streamlit web dashboard to perform real-time model evaluation without retraining.

---

## b. Dataset Description
* **Dataset Name**: Heart Failure Prediction Dataset (`heart.csv`)
* **Source**: UCI Machine Learning Repository / Kaggle
* **Instance Size**: 918 patient observations (Test partition: 184 samples, 20% stratified split)
* **Feature Dimensionality**: 11 raw clinical attributes expanded to 15 features post One-Hot Encoding (satisfies the minimum 12 feature criterion)
* **Target Label**: `HeartDisease` (Binary: `0` = No Heart Disease, `1` = Heart Disease)

### Attribute Information:
1. `Age`: Patient age in years [numeric]
2. `Sex`: Biological sex of the patient [`M`: Male, `F`: Female]
3. `ChestPainType`: Type of chest pain [`TA`: Typical Angina, `ATA`: Atypical Angina, `NAP`: Non-Anginal Pain, `ASY`: Asymptomatic]
4. `RestingBP`: Resting blood pressure in mm Hg [numeric]
5. `Cholesterol`: Serum cholesterol in mg/dl [numeric]
6. `FastingBS`: Fasting blood sugar level [`1`: if FastingBS > 120 mg/dl, `0`: otherwise]
7. `RestingECG`: Resting electrocardiogram results [`Normal`, `ST`: having ST-T wave abnormality, `LVH`: showing probable or definite left ventricular hypertrophy]
8. `MaxHR`: Maximum heart rate achieved during exercise [numeric, 60–202]
9. `ExerciseAngina`: Exercise-induced angina [`Y`: Yes, `N`: No]
10. `Oldpeak`: ST depression induced by exercise relative to rest [numeric]
11. `ST_Slope`: The slope of the peak exercise ST segment [`Up`: Upsloping, `Flat`: Flat, `Down`: Downsloping]

---

## d. Models Used

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8859 | 0.9308 | 0.8716 | 0.9314 | 0.9005 | 0.7694 |
| **Decision Tree** | 0.7989 | 0.8194 | 0.8155 | 0.8235 | 0.8195 | 0.5926 |
| **KNN** | 0.8913 | 0.9434 | 0.8942 | 0.9118 | 0.9029 | 0.7797 |
| **Naive Bayes** | 0.9130 | 0.9451 | 0.9300 | 0.9118 | 0.9208 | 0.8246 |
| **Support Vector Machine** | 0.8859 | 0.9464 | 0.8857 | 0.9118 | 0.8986 | 0.7686 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed consistently well across the linear boundary with scaled features, achieving 0.8859 accuracy. It attained the highest sensitivity/recall score (0.9314), correctly identifying 95 out of 102 heart disease cases in the test set. |
| **Decision Tree** | Produced the lowest overall scores (0.7989 accuracy, 0.5926 MCC). The axis-aligned rectangular splits struggled to model continuous cardiac thresholds without overfitting the training partition. |
| **KNN** | Scored 0.8913 accuracy and 0.9434 AUC ($k=7$). Feature standardization via `StandardScaler` was essential in preventing higher-magnitude attributes (`Cholesterol`, `RestingBP`) from distorting Euclidean distance calculations. |
| **Naive Bayes** | Delivered the highest overall accuracy (0.9130), precision (0.9300), F1 score (0.9208), and MCC (0.8246). Despite the feature independence assumption, Gaussian Naive Bayes showed robust classification on encoded diagnostic indicators. |
| **Support Vector Machine** | Achieved the highest AUC score (0.9464) using an RBF kernel, indicating strong discriminative capability across confidence thresholds while maintaining balanced precision (0.8857) and recall (0.9118). |
| **Overall Winner for your dataset?** | **Naive Bayes (GaussianNB)** is the overall winner. It achieved the highest overall Accuracy (91.30%), Precision (93.00%), F1 Score (0.9208), and Matthews Correlation Coefficient (0.8246), successfully balancing false-positive suppression with reliable positive identification in a clinical setting. |

---

## Setup and Local Execution Instructions

### 1. Clone the repository:
```bash
git clone [https://github.com/](https://github.com/)<your-github-username>/<your-repo-name>.git
cd project-folder

2.Install dependencies:
pip install -r requirements.txt
3. Train models and generate model binaries:
python train_models.py
4. Run the Streamlit web application:
streamlit run app.py
