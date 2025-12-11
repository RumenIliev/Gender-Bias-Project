import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree


# income - gender + education + occupation
def fit_logistic_regression(X, y):
    # Adding intercept explicitly
    X_with_intercept = sm.add_constant(X)

    model = sm.Logit(y, X_with_intercept)
    result = model.fit(disp=False)

    return result

# Extracts coefficient, p-value and confidence interval for the gender variable.
def extract_gender_effect(model_result):
    coef = model_result.params["gender"]
    p_value = model_result.pvalues["gender"]

    ci_lower, ci_upper = model_result.conf_int().loc["gender"]

    odds_ratio = np.exp(coef)

    info = {
        "coef": coef,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": odds_ratio,
    }

    return info


# Decision Tree
def evaluate_decision_tree(X, y, max_depth=None, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    return {"model": model, "accuracy": accuracy}


# Plot results
def plot_decision_tree(model, feature_names, max_depth=3):
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["<=50K", ">50K"],
        filled=True,
        rounded=True,
        max_depth=max_depth
    )
    plt.show()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)

    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()
