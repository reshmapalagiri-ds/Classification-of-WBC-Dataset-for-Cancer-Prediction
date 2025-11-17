## 🧬 Classification of WBC Dataset for Cancer Prediction
A Machine Learning Project for Breast Cancer Diagnosis

## 📌 Project Overview

This project applies machine learning classification techniques to the Breast Cancer Wisconsin (WBC) dataset to distinguish between Malignant (M) and Benign (B) tumors.
The focus is on building an interpretable Decision Tree Classifier, evaluating its performance, visualizing its decision-making process, and comparing it with Logistic Regression for better understanding.

This project is ideal for learning:

Data preprocessing

Decision tree modeling

Overfitting vs underfitting

Feature importance

Model visualization in ML

Interpretable machine learning for medical data

## 📂 Dataset Description

The dataset (wbc.csv) contains 30 numerical features extracted from digitized images of breast cell nuclei.
These include measurements such as:

- Radius

- Texture

- Perimeter

- Area

- Smoothness

- Concavity

- Symmetry

- Fractal dimension

- Target Labels

   M → Malignant → 1

   B → Benign → 0

## Preprocessing Steps

- Dropped irrelevant columns: id, Unnamed: 32

- Removed duplicate entries

- Checked for missing values

- Mapped diagnosis labels

- Split into train/test sets

## 🔧 Workflow Overview
1️⃣ Data Preprocessing

Handle missing data

Drop unnecessary columns

Encode target variable

Split data into training and testing

2️⃣ Model Training (Decision Tree)

A base Decision Tree model is trained using:

DecisionTreeClassifier()

3️⃣ Model Evaluation

Metrics used:

Accuracy Score

F1 Score

Evaluated on the test dataset to prevent overfitting.

4️⃣ Hyperparameter Tuning

To redu­ce overfitting, various tree depths are tested (1–20) using 10-fold cross-validation:

cross_val_score(dt, x_train, y_train, cv=10)

5️⃣ Visualization

Includes:

Decision boundary visualization (Decision Tree vs Logistic Regression)

Feature importance ranking

Full decision tree plot

Probability predictions

These visualizations help understand model interpretability and decision-making.

## 🎨 Feature Importance

The project analyzes which features contribute most to the classification, helping identify medically important predictors.

## 🌳 Decision Tree Visualization

The full tree is plotted using:

tree.plot_tree(dt, filled=True)


This provides insights into:

Split rules

Gini impurity

Class distribution

## 🧪 Prediction Example

The trained model predicts the tumor class for a sample from the test set using:

dt.predict()
dt.predict_proba()

## 📸 Screenshots

Below are screenshots from the project for better understanding:

1️⃣ Dataset Preview
(showing first few columns due to dataset width)
![Dataset Prreview](https://github.com/reshmapalagiri-ds/Classification-of-WBC-Dataset-for-Cancer-Prediction/blob/main/WBC%20dataset%20Preview.png?raw=true)

2️⃣ Model Performance Output
![Model Performnace output](https://github.com/reshmapalagiri-ds/Classification-of-WBC-Dataset-for-Cancer-Prediction/blob/main/Model%20Performnance%20output.png?raw=true)

3️⃣ Decision Boundary Visualization
![Decision Boundary Visualization](https://github.com/reshmapalagiri-ds/Classification-of-WBC-Dataset-for-Cancer-Prediction/blob/main/Visualisation%20boundary.png?raw=true)

4️⃣ Decision Tree Structure
![Decision Tree Structure](https://github.com/reshmapalagiri-ds/Classification-of-WBC-Dataset-for-Cancer-Prediction/blob/main/Decision%20Tree%20Structure.png?raw=true)


## 🛠️ Technologies & Libraries

Python

Pandas

NumPy

Scikit-learn

Matplotlib

mlxtend

Google Colab

## ▶️ How to Run the Project
1. Clone the Repository
git clone https://github.com/reshmapalagiri-ds/Classiication-of-WBC-Dataset-for-Cancer-Prediction.git

cd Classification-of-WBC-Dataset-for-Cancer-Prediction

2. Install Dependencies
   
pip install -r requirements.txt

3. Run the Notebook

Open the .ipynb file using:

Google Colab

Jupyter Notebook

VS Code Notebook

## 📁 Project Structure
|-- wbc.csv
|-- DT-wbc-project.ipynb
|-- README.md
|-- imagename.png
|-- screenshots/
|     |-- dataset_preview.png
|     |-- model_performance.png
|     |-- decision_boundary.png
|     |-- decision_tree.png
|-- requirements.txt

## 💡 Key Learnings

✔ Understanding Decision Trees
✔ Handling overfitting via depth tuning
✔ Visualizing classifier behavior
✔ Extracting medically relevant feature importance
✔ Building explainable ML models

## 🔮 Future Enhancements

Possible improvements:

Add Random Forest & Gradient Boosting

Plot ROC–AUC curves

Use SHAP values for deeper interpretability

Hyperparameter tuning with GridSearchCV

Deploy using Streamlit or Flask

## 📬 Contact

📧 Email: reshmapalagiri1807@gmail.com

🔗 LinkedIn: www.linkedin.com/in/palagiri-reshma
