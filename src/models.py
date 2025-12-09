import numpy as np
import statsmodels.api as sm


# income ~ gender + education + occupation
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
