import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load the training and testing datasets
train_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")
test_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")

# Assuming 'Label' is the name of the column containing the target labels
X_train = train_df.drop(columns=['Label'])  # Features for training
y_train = train_df['Label']  # Target labels for training

X_test = test_df.drop(columns=['Label'])  # Features for testing
y_test = test_df['Label']  # Target labels for testing

# Initialize models
lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)

# Hyperparameter tuning for Logistic Regression
param_grid_lr = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(lr_model, param_grid_lr, cv=5)
lr_grid.fit(X_train, y_train)

# Train models
lr_model = lr_grid.best_estimator_
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_lr = lr_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# Evaluate model performance
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Print accuracy scores
print("Logistic Regression Accuracy:", accuracy_lr)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Decision Tree Accuracy:", accuracy_dt)

# Calculate metrics for Logistic Regression
precision_lr, recall_lr, f1_score_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='weighted')
auc_roc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test), multi_class='ovr')

# Calculate metrics for Naive Bayes
precision_nb, recall_nb, f1_score_nb, _ = precision_recall_fscore_support(y_test, y_pred_nb, average='weighted')
auc_roc_nb = roc_auc_score(y_test, nb_model.predict_proba(X_test), multi_class='ovr')

# Calculate metrics for Decision Tree
precision_dt, recall_dt, f1_score_dt, _ = precision_recall_fscore_support(y_test, y_pred_dt, average='weighted')
auc_roc_dt = roc_auc_score(y_test, dt_model.predict_proba(X_test), multi_class='ovr')

# Print metrics for each model
print("Logistic Regression Metrics:")
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1-score:", f1_score_lr)
print("AUC-ROC:", auc_roc_lr)
print()

print("Naive Bayes Metrics:")
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("F1-score:", f1_score_nb)
print("AUC-ROC:", auc_roc_nb)
print()

print("Decision Tree Metrics:")
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-score:", f1_score_dt)
print("AUC-ROC:", auc_roc_dt)
