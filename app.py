import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    """Preprocess dataset: handle missing values, encode categorical features, and scale numeric data."""
    df.fillna(method='ffill', inplace=True)  # Forward-fill missing values
    df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(lambda col: pd.Categorical(col).codes)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """Perform grid search with parallel processing to find the best model parameters."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X, y, is_classification):
    """Train and evaluate model performance based on its type (classification/regression)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification:
        return {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
        }
    else:
        return {
            'MAE': round(mean_absolute_error(y_test, y_pred), 2),
            'MSE': round(mean_squared_error(y_test, y_pred), 2),
            'R2 Score': round(r2_score(y_test, y_pred), 2)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type. Only CSV files are allowed!"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    X, y = preprocess_data(df)

    is_classification = y.nunique() < 20 and np.issubdtype(y.dtype, np.integer)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}) if is_classification else None,
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}) if is_classification else (DecisionTreeRegressor(), {'max_depth': [3, 5, 10]}),
        'Random Forest': (RandomForestClassifier(n_jobs=-1), {'n_estimators': [50, 100, 150]}) if is_classification else (RandomForestRegressor(n_jobs=-1), {'n_estimators': [50, 100, 150]}),
        'Support Vector Machine': (SVC(), {'C': [0.1, 1, 10]}) if is_classification else (SVR(), {'C': [0.1, 1, 10]}),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}) if is_classification else (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
        'Naive Bayes': (GaussianNB(), {}) if is_classification else None,
        'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [50, 100]}) if is_classification else None
    }

    results = []
    for model_name, model_data in models.items():
        if model_data:
            model, param_grid = model_data
            best_model = hyperparameter_tuning(model, param_grid, X, y) if param_grid else model
            metrics = evaluate_model(best_model, X, y, is_classification)
            results.append({'model': model_name, **metrics})

    # Sort results based on accuracy (classification) or R2 score (regression)
    results.sort(key=lambda x: x['accuracy' if is_classification else 'R2 Score'], reverse=True)

    best_model = results[0] if results else None

    return render_template('results.html', results=results, best_model=best_model)

if __name__ == '__main__':
    app.run(debug=True)
