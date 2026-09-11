📡 Telco Customer Churn Prediction

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white">{=html}
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white">{=html}
<img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white">{=html}
<img src="https://img.shields.io/badge/SMOTE-Imbalanced%20Learning-orange">{=html}
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">{=html}
<img src="https://img.shields.io/badge/Status-Completed-success">{=html}

</p>

<p align="center">

<b>{=html}End-to-end Machine Learning project to predict telecom
customer churn.</b>{=html}

</p>

📌 Project Overview

This project predicts whether a telecom customer is likely to Churn
or Not Churn using customer service, contract, tenure, payment, and
billing information.

The complete workflow implemented in the notebook is covered below from
Step 2 to Step 8.



📑 Table of Contents


Step 3 - Data Preprocessing & Feature
Engineering

Step 4 - Model Building: KNN & Naive
Bayes

Step 5 - Model Building: SVM & Decision
Tree

Step 6 - Model Evaluation &
Comparison

Step 7 - Error Analysis &
Interpretation

Step 8 - Final Pipeline, Model Saving &
Prediction

Project Screenshots

Project Demo GIF

Project Structure

Installation & Usage

Business Objective

Skills Demonstrated

Future Improvements

                                
======Step 2 - Dataset Loading & Exploratory Data Analysis======
                                

======2.1 Load & Inspect======


The Telco Customer Churn CSV dataset is loaded using Pandas.

The notebook checks:

Dataset shape

Dataset information

Descriptive statistics

First 10 rows

TotalCharges data type

Churn class distribution

Churn percentage

Dataset

Property       Value

Rows           7,043
Columns           21
Target       Churn

Target Variable

Yes → Customer churned

No → Customer did not churn

TotalCharges Cleaning

TotalCharges is converted from text/object to numeric data.

Blank values are converted to missing values and then filled with 0 in
this step.

Churn Distribution

The notebook checks:

df["Churn"].value_counts()

and calculates the percentage distribution using:

df["Churn"].value_counts(normalize=True) * 100

📷 Screenshot

Add the Step 2 notebook/output screenshot here:

![Step 2 - Dataset Loading and EDA](docs/screenshots/step2-eda.png)


======Step 3 - Data Preprocessing & Feature Engineering======


Step 3 prepares the dataset for machine learning.


======3.1 Drop & Clean======


Drop Customer ID

customerID is removed because it is an identifier and is not required
as a predictive feature.

Convert TotalCharges

TotalCharges is converted into numeric format.

Missing values are filled using the median:

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

The notebook checks:

Data type

Null values

Dataset shape


======3.2 Feature Engineering======


Three additional features are created.

tenure_group

Customers are divided into:

Group    Tenure

New      0-12 months
Mid      13-36 months
Senior   37-60 months
Loyal    61-72 months

num_services

Counts the number of subscribed services from:

Online Security

Online Backup

Device Protection

Tech Support

Streaming TV

Streaming Movies

AutoPay

A binary feature is created from PaymentMethod.

Automatic payment → 1

Other payment method → 0


======3.3 Encoding======


Binary Encoding

The following Yes/No columns are converted to 1/0:

Partner

Dependents

PhoneService

PaperlessBilling

Churn

One-Hot Encoding

The following categorical columns are one-hot encoded:

InternetService

Contract

PaymentMethod

Tenure Group Encoding

New    → 0
Mid    → 1
Senior → 2
Loyal  → 3


======3.4 Train-Test Split & Scaling======


The dataset is divided into:

80% Training

20% Testing

The split uses:

random_state=42
stratify=y

Selected numerical columns are standardized using StandardScaler:

tenure

MonthlyCharges

TotalCharges

num_services


======3.5 Handle Class Imbalance======


The project uses SMOTE to handle class imbalance.

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

SMOTE is applied to the training data only.

The notebook compares class distribution:

Before SMOTE
      ↓
After SMOTE

This helps give the minority churn class more representation during
training.


======3.6 Train-Test Split======


The notebook also performs an 80/20 train-test split using:

train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

📷 Screenshot

![Step 3 - Preprocessing](docs/screenshots/step3-preprocessing.png)


======Step 4 - Model Building: KNN & Naive Bayes======


======4.1 KNN Baseline======


A K-Nearest Neighbors classifier is trained with:

KNeighborsClassifier(n_neighbors=5)

The baseline KNN is evaluated using:

Accuracy

Precision

Recall

F1 Score

AUC-ROC


======4.2 KNN Hyperparameter Tuning======


The notebook tests:

k = 1, 3, 5, 7, 9, 11, 15

For each k, F1 Score is calculated.

The best value is selected automatically:

best_k = k_values[f1_scores.index(max(f1_scores))]

📊 K vs F1 Score

The notebook creates a line chart showing the relationship between K
value and F1 Score.

![KNN K vs F1](docs/screenshots/knn-k-vs-f1.png)


======4.3 Retrain KNN with Optimal K======


The KNN model is retrained using the selected best_k.

A confusion matrix is generated to show:

True Negatives

False Positives

False Negatives

True Positives

![KNN Confusion Matrix](docs/screenshots/knn-confusion-matrix.png)


======4.4 KNN ROC Curve & AUC======


The project calculates:

ROC Curve

AUC-ROC

The ROC curve is plotted for the optimized KNN model.

![KNN ROC Curve](docs/screenshots/knn-roc.png)

Note: The notebook imports GaussianNB for the Naive Bayes model
and uses it in the Step 6 four-model comparison. The uploaded notebook
does not contain a separate standalone Naive Bayes training subsection
in Step 4.


======Step 5 - Model Building: SVM & Decision Tree======


======5.1 Support Vector Machine (SVM)======


The baseline SVM uses:

SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    probability=True,
    random_state=42
)

The model is evaluated using:

Accuracy

Precision

Recall

F1 Score

AUC-ROC

Confusion Matrix

ROC Curve

SVM C Tuning

The following values are tested:

C = 0.1, 1, 10, 100

Three-fold cross-validation is used with F1 scoring.

The best C is selected automatically.

best_C = C_values[f1_scores.index(max(f1_scores))]

📊 C vs F1 Score

![SVM C vs F1](docs/screenshots/svm-c-vs-f1.png)

Final SVM

The SVM is retrained using the best C.

Final evaluation includes:

Accuracy

Precision

Recall

F1 Score

AUC-ROC

ROC Curve

![SVM ROC Curve](docs/screenshots/svm-roc.png)


======5.2 Decision Tree Classifier======


The baseline Decision Tree uses:

DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

The model is evaluated using:

Accuracy

Precision

Recall

F1 Score

AUC-ROC

Confusion Matrix

Decision Tree Visualization

The first three levels of the Decision Tree are visualized.

![Decision Tree](docs/screenshots/decision-tree.png)

The notebook also identifies the root split feature.

Decision Tree max_depth Tuning

The notebook tests:

3, 4, 5, 6, 7, 8, None

Five-fold cross-validation is used with F1 scoring.

The best depth is selected automatically.

📊 Depth vs F1 Score

![Decision Tree Depth vs F1](docs/screenshots/dt-depth-vs-f1.png)

Final Decision Tree

The Decision Tree is retrained using the best depth.

Feature importance is calculated and the Top 15 features are displayed.

![Top 15 Feature Importance](docs/screenshots/top15-feature-importance.png)

SMOTE vs class_weight='balanced'

The project compares Decision Tree recall using:

SMOTE

class_weight="balanced"

The approach with higher Recall is identified as the better approach
based on the notebook's comparison.

Step 6 - Model Evaluation & Comparison

Four machine learning models are compared:

Model

KNN
Naive Bayes
SVM
Decision Tree

Each model is trained using the SMOTE training data.

Evaluation Metrics

The comparison includes:

Accuracy

Precision

Recall

F1-Score

AUC-ROC

Training Time

The results are stored in a Pandas DataFrame called:

comparison

The results are sorted by Recall in descending order.

Why Recall?

For churn prediction, identifying customers who actually churn is
important.

Therefore, Recall is used by the notebook to automatically identify the
best model:

best_model_row = comparison.iloc[0]

📊 Model Comparison

![Model Comparison](docs/screenshots/model-comparison.png)

📈 ROC Curves - All 4 Models

ROC curves are generated for:

KNN

Naive Bayes

SVM

Decision Tree

![ROC Curves](docs/screenshots/all-model-roc.png)


📊 Precision vs Recall


A grouped bar chart compares Precision and Recall for all four models.

![Precision vs Recall](docs/screenshots/precision-vs-recall.png)

🏆 Best Model

The notebook automatically prints:

Best Model
Recall
Precision
F1-Score
AUC-ROC

The best model is selected based on the highest Recall.


======Step 7 - Error Analysis & Interpretation======


Step 7 performs detailed error analysis on the best model.


======7.1 Best Model Prediction======


The best model is selected using the highest Recall.

The test set is then predicted using that model.


======7.2 False Negatives======


False Negatives are customers who:

Actual = Churn
Predicted = No Churn

The notebook calculates:

fn_mask = (y_test == 1) & (y_pred_best == 0)

and reports the total number of False Negatives.


======7.3 False Negative Customer Profile======


For False Negative customers, the project analyzes:

Tenure

Monthly Charges

Contract type

![False Negative Analysis](docs/screenshots/false-negative-analysis.png)


======7.4 Overall Churner Profile======


The False Negative profile is compared with all actual churners.

The notebook compares:

Average tenure

Average MonthlyCharges

Contract distribution

Contract percentages


======7.5 Error Analysis Pattern======


The notebook reports the observed pattern:

False Negatives have higher average tenure than overall churners.

False Negatives have lower average MonthlyCharges.

One-year contracts appear prominently among missed customers.

This analysis can help identify areas for future model improvement.

💡 Business Interpretation

The error analysis suggests that some customers with longer tenure and
lower monthly charges can be harder for the model to identify as
churners.

These customers may require additional features or improved
probability-threshold tuning in future versions.

======Step 7.2 - Feature Importance & Business Interpretation======

The Decision Tree feature importance is calculated.

The project extracts the Top 5 Decision Tree split features.

For each feature, the notebook prints:

Feature name

Importance score

Business-friendly explanation

![Top 5 Features](docs/screenshots/top5-features.png)

Optional SVM Permutation Importance

The notebook also calculates permutation importance for the SVM.

The Top 10 features are displayed and visualized.

![SVM Feature Importance](docs/screenshots/svm-feature-importance.png)


======Step 8 - Final Pipeline, Model Saving & Prediction======


Step 8 creates the final reusable machine learning pipeline.


======8.1 Create Original X and y======


The original dataframe is separated into:

X_original → Features
y_original → Churn target


======8.2 Train-Test Split======


The original customer data is split into:

80% training

20% testing

with:

random_state=42
stratify=y_original


======8.3 Identify Numerical & Categorical Columns======


The pipeline identifies:

Numerical Columns

Columns with:

int64
float64

Categorical Columns

Columns with:

object


======8.4 Create Preprocessor======


The project uses:

ColumnTransformer

with:

OneHotEncoder(handle_unknown="ignore")

This allows the pipeline to process categorical features automatically.


======8.5 Create Final Pipeline======


The final pipeline combines:

Preprocessor
      ↓
Final Decision Tree

The model used is final_dt.


======8.6 Train Pipeline======


The pipeline is trained using the original training data:

pipeline.fit(
    X_train_original,
    y_train_original
)


======8.7 Save Model======


The trained pipeline is saved as:

churn_model.pkl

using Joblib:

joblib.dump(
    pipeline,
    "churn_model.pkl"
)


======8.8 Load Model======


The saved model is loaded again:

loaded_pipeline = joblib.load(
    "churn_model.pkl"
)


======8.9 Select 5 Original Customers======


Five customers from the original test dataset are selected for
prediction.


======8.10 Predict Churn Probability======


The pipeline predicts:

Churn Probability

for each selected customer.


======8.11 Predict Final Label======


The pipeline predicts:

Churn

or:

No Churn


======8.12 Final Output======


The notebook prints for five customers:

Customer 1
Predicted Probability: ...
Final Label: Churn / No Churn

📷 Final Prediction Screenshot

![Final Churn Prediction](docs/screenshots/final-prediction.png)

📸 Project Screenshots

Recommended GitHub screenshot organization:

docs/
└── screenshots/
    ├── step2-eda.png
    ├── step3-preprocessing.png
    ├── knn-k-vs-f1.png
    ├── knn-confusion-matrix.png
    ├── knn-roc.png
    ├── svm-c-vs-f1.png
    ├── svm-roc.png
    ├── decision-tree.png
    ├── dt-depth-vs-f1.png
    ├── top15-feature-importance.png
    ├── model-comparison.png
    ├── all-model-roc.png
    ├── precision-vs-recall.png
    ├── false-negative-analysis.png
    ├── top5-features.png
    ├── svm-feature-importance.png
    └── final-prediction.png

🎥 Project Demo GIF

Add a short GIF showing the complete notebook workflow.

Recommended location:

docs/
└── demo/
    └── telco-churn-demo.gif

Add it to this README:

![Telco Customer Churn Demo](docs/demo/telco-churn-demo.gif)

Suggested GIF Flow

Step 2 → Dataset & EDA
        ↓
Step 3 → Preprocessing
        ↓
Step 4 → KNN
        ↓
Step 5 → SVM + Decision Tree
        ↓
Step 6 → Model Comparison
        ↓
Step 7 → Error Analysis
        ↓
Step 8 → Final Prediction

💡 Keep the GIF preferably below 10 MB for easy GitHub viewing.

📁 Project Structure

Telco-Customer-Churn/
│
├── practical(2).ipynb
├── Telco-Customer-Churn.csv
├── churn_model.pkl
├── README.md
│
└── docs/
    ├── screenshots/
    │   ├── step2-eda.png
    │   ├── step3-preprocessing.png
    │   ├── knn-k-vs-f1.png
    │   ├── knn-confusion-matrix.png
    │   ├── knn-roc.png
    │   ├── svm-c-vs-f1.png
    │   ├── svm-roc.png
    │   ├── decision-tree.png
    │   ├── dt-depth-vs-f1.png
    │   ├── top15-feature-importance.png
    │   ├── model-comparison.png
    │   ├── all-model-roc.png
    │   ├── precision-vs-recall.png
    │   ├── false-negative-analysis.png
    │   ├── top5-features.png
    │   ├── svm-feature-importance.png
    │   └── final-prediction.png
    │
    └── demo/
        └── telco-churn-demo.gif

🚀 Installation & Usage

1. Clone Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Telco-Customer-Churn

2. Install Required Libraries

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter

3. Open Jupyter Notebook

jupyter notebook

Open:

practical(2).ipynb

4. Keep Dataset in Project Folder

Make sure:

Telco-Customer-Churn.csv

is available in the notebook's working directory.

5. Run the Notebook

Run the notebook cells from Step 2 through Step 8 in order.

🎯 Business Objective

Customer churn can negatively affect telecom companies through lost
recurring revenue and increased customer-acquisition costs.

A churn prediction system can help a company:

Identify customers at risk of leaving.

Prioritize retention campaigns.

Understand customer behavior.

Analyze contract and service patterns.

Reduce customer loss.

Support data-driven decisions.

🧠 Skills Demonstrated

Python

Pandas

NumPy

Exploratory Data Analysis

Data Cleaning

Feature Engineering

Binary Encoding

One-Hot Encoding

Train-Test Split

StandardScaler

SMOTE

KNN

Naive Bayes

SVM

Decision Tree

Cross Validation

Hyperparameter Tuning

Accuracy

Precision

Recall

F1 Score

AUC-ROC

Confusion Matrix

ROC Curves

Error Analysis

Feature Importance

Permutation Importance

ColumnTransformer

Scikit-learn Pipeline

Joblib Model Saving

Churn Probability Prediction

🔮 Future Improvements

Add cross-validation based final model selection.

Tune the churn probability threshold.

Add SHAP explainability.

Build a Streamlit dashboard.

Create an API for real-time prediction.

Add automated model monitoring.

Compare additional ensemble models.

Add model calibration.

Deploy the final model to a cloud platform.

👨‍💻 Author

Rahul Zala

Machine Learning / Data Science Project

⭐ Project Status

Completed --- Machine Learning Classification Project

The project covers the complete workflow from Step 2: Dataset Loading
& EDA to Step 8: Final Pipeline, Model Saving & Churn Prediction.

If you find this project useful, consider giving the repository a ⭐.
