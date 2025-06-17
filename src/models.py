import joblib
import shap
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Classification extensions
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_models(X_train, y_train):
    lr = LinearRegression()
    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(random_state=42)
    xg = xgb.XGBRegressor(random_state=42)
    for model in (lr, dt, rf, xg):
        model.fit(X_train, y_train)
    return {'LinearRegression': lr, 'DecisionTree': dt, 'RandomForest': rf, 'XGBoost': xg}


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2, y_pred


def plot_metrics(results):
    """
    results: dict of model_name -> (mae, mse, r2, y_pred)
    """
    models = list(results.keys())
    mae = [results[m][0] for m in models]
    mse = [results[m][1] for m in models]
    r2  = [results[m][2] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(models, mae, color='skyblue'); axes[0].set_title('MAE'); axes[0].tick_params(axis='x', rotation=45)
    axes[1].bar(models, mse, color='lightgreen'); axes[1].set_title('MSE'); axes[1].tick_params(axis='x', rotation=45)
    axes[2].bar(models, r2, color='salmon'); axes[2].set_title('R²'); axes[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def plot_decision_tree(model, feature_names):
    """
    Plot the first tree of a RandomForest or a standalone DecisionTree.
    """
    tree = None
    if hasattr(model, 'estimators_'):
        tree = model.estimators_[0]
    else:
        tree = model
    plt.figure(figsize=(20,10))
    plot_tree(tree, feature_names=feature_names, filled=True, max_depth=3, fontsize=10)
    plt.show()

def plot_shap_summary(model, X, max_display=15):
    """
    Generate a SHAP summary plot showing the top features influencing model predictions.
    Only works with tree-based models or models supported by SHAP.
    """
    # Create TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)
    plt.show()


def train_classification_models(X_train, y_train):
    """
    Train a set of classification models: Logistic Regression, Decision Tree, Random Forest.
    """
    lr = LogisticRegression(max_iter=1000)
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    models = {'LogisticRegression': lr, 'DecisionTree': dt, 'RandomForest': rf}
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate classification model, returning accuracy, precision, recall, f1, roc_auc, and predictions.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    return acc, prec, rec, f1, roc_auc, y_pred, y_proba

def plot_shap_summary(model, X, max_display=15):
    """
    Generate a SHAP summary plot showing the top features influencing model predictions.
    Supports tree-based models via TreeExplainer and linear models via LinearExplainer.
    """
    # Choose explainer based on model type
    try:
        if hasattr(model, 'feature_perturbation'):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)
        plt.show()
    except Exception as e:
        print(f"❌ SHAP plotting error: {e} Please ensure the model and data are compatible with SHAP.")