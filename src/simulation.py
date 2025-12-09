import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Logistic function


def simulate_data(n_samples, bias=False, seed=0):
    np.random.seed(seed)

    genders = []
    educations = []
    occupations = []
    incomes = []

    education_levels = ["low", "medium", "high"]
    occupation_levels = ["manual", "service", "professional"]

    for i in range(n_samples):
        #Gender
        if np.random.rand() < 0.5:
            gender = "Male"
            g = 0
        else:
            gender = "Female"
            g = 1

        # Education | Gender
        if gender == "Male":
            edu_probs = [0.35, 0.40, 0.25]
        else:
            edu_probs = [0.25, 0.45, 0.30]

        education = np.random.choice(education_levels, p=edu_probs)
        e = education_levels.index(education)

        # Occupation | Gender, Education
        if education == "low":
            if gender == "Male":
                occ_probs = [0.65, 0.25, 0.10]
            else:
                occ_probs = [0.45, 0.35, 0.20]
        elif education == "medium":
            if gender == "Male":
                occ_probs = [0.45, 0.25, 0.30]
            else:
                occ_probs = [0.30, 0.30, 0.40]
        else:
            if gender == "Male":
                occ_probs = [0.20, 0.20, 0.60]
            else:
                occ_probs = [0.10, 0.20, 0.70]

        occupation = np.random.choice(occupation_levels, p=occ_probs)
        o = occupation_levels.index(occupation)

        # Income
        # Logistic model
        intercept = -2.0
        edu_effects = [0.0, 0.8, 1.6]
        occ_effects = [0.0, 0.6, 1.2]

        gender_effect = 0.0
        if bias:
            gender_effect = -0.7  # disadvantage for women

        linear_score = (
            intercept
            + edu_effects[e]
            + occ_effects[o]
            + gender_effect * g
        )

        prob_income = sigmoid(linear_score)
        income = 1 if np.random.rand() < prob_income else 0

        # Store results
        genders.append(gender)
        educations.append(education)
        occupations.append(occupation)
        incomes.append(income)

    df = pd.DataFrame(
        {
            "gender": genders,
            "education": educations,
            "occupation": occupations,
            "income": incomes,
        }
    )

    return df