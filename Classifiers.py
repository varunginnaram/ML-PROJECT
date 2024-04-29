



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load the training and testing datasets
train_df = pd.read_excel("C:\\Users\\varun\\OneDrive\\Desktop\\ml\\Abstractive_Embeddings_Fasttext_Hindi (1).xlsx")
test_df = pd.read_excel("C:\\Users\\varun\\OneDrive\\Desktop\\ml\\Test_Abstractive_Hindi.xlsx")

# Assuming 'target_column' is the name of the column containing the target labels
X_train = train_df.drop(columns=['Label'])  # Features for training
y_train = train_df['Label']  # Target labels for training

X_test = test_df.drop(columns=['Label'])  # Features for testing
y_test = test_df['Label']  # Target labels for testing

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize models
svm_model = SVC(kernel='linear', probability=True)
rf_model = RandomForestClassifier()
gbm_model = GradientBoostingClassifier()
xgb_model = XGBClassifier()
catboost_model = CatBoostClassifier()
adaboost_model=AdaBoostClassifier()

# Train models
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)
adaboost_model.fit(X_train,y_train)


# Make predictions on the validation set
y_pred_svm_val = svm_model.predict(X_val)
y_pred_rf_val = rf_model.predict(X_val)
y_pred_gbm_val = gbm_model.predict(X_val)
y_pred_xgb_val = xgb_model.predict(X_val)
y_pred_catboost_val = catboost_model.predict(X_val)
y_pred_adaboost_val=adaboost_model.predict(X_val)

# Make predictions on the testing set
y_pred_svm_test = svm_model.predict(X_test)
y_pred_rf_test = rf_model.predict(X_test)
y_pred_gbm_test = gbm_model.predict(X_test)
y_pred_xgb_test = xgb_model.predict(X_test)
y_pred_catboost_test = catboost_model.predict(X_test)
y_pred_adaboost_test=adaboost_model.predict(X_test)
# Evaluate models on validation set
def evaluate_model(y_true, y_pred, model, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_true, model.predict_proba(X_val), multi_class='ovr')
    
    print(f"Metrics for {model_name} on Validation Set:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
    print("AUC-ROC:", auc_roc)
    print()

evaluate_model(y_val, y_pred_svm_val, svm_model, "SVM")
evaluate_model(y_val, y_pred_rf_val, rf_model, "Random Forest")
evaluate_model(y_val, y_pred_gbm_val, gbm_model, "Gradient Boosting")
evaluate_model(y_val, y_pred_xgb_val, xgb_model, "XGBoost")
evaluate_model(y_val, y_pred_catboost_val, catboost_model, "CatBoost")
evaluate_model(y_val,y_pred_adaboost_val,adaboost_model, "AdaBoost")

# Evaluate models on testing set
def evaluate_model_test(y_true, y_pred, model, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_true, model.predict_proba(X_test), multi_class='ovr')
    
    print(f"Metrics for {model_name} on Testing Set:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
    print("AUC-ROC:", auc_roc)
    print()

evaluate_model_test(y_test, y_pred_svm_test, svm_model, "SVM")
evaluate_model_test(y_test, y_pred_rf_test, rf_model, "Random Forest")
evaluate_model_test(y_test, y_pred_gbm_test, gbm_model, "Gradient Boosting")
evaluate_model_test(y_test, y_pred_xgb_test, xgb_model, "XGBoost")
evaluate_model_test(y_test, y_pred_catboost_test, catboost_model, "CatBoost")
evaluate_model_test(y_test, y_pred_adaboost_test, adaboost_model, "AdaBoost")
