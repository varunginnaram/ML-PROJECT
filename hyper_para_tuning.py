import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint

# Load training and testing data
train_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")  # Replace with your train dataset path
test_data = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")    # Replace with your test dataset path

# Split data into features and target
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']
# Hyperparameter grids for different classifiers
classifier_grid = {
    'Perceptron': {
        'alpha': uniform(0.0001, 0.01),
        'max_iter': randint(300, 600),
    },
    'LogisticRegression': {
        'C': uniform(0.01, 10),
        'solver': ['liblinear', 'newton-cg'],
    },
    'KNeighborsClassifier': {
        'n_neighbors': randint(1, 20),
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    },
    'SVC': {
        'C': uniform(0.01, 10),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': uniform(0.001, 0.1),  # Adjust for non-linear kernels
    },
    'DecisionTreeClassifier': {
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'criterion': ['gini', 'entropy'],
    },
    'RandomForestClassifier': {
        'n_estimators': randint(10, 100),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
    }
}

# Perform RandomizedSearchCV for each classifier
for classifier, param_grid in classifier_grid.items():
    classifier_object = eval(classifier)()  # Get classifier object dynamically
    clf = RandomizedSearchCV(classifier_object, param_grid, n_iter=10, cv=5, random_state=42)
    clf.fit(X_train, y_train)

    # Print best parameters and best scores
    print(f"{classifier} - Best Parameters:", clf.best_params_)


