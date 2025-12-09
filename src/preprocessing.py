import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_dataframe(df):
    # Basic sanity checks
    required_columns = ["gender", "education", "occupation", "income"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Encoding gender
    gender_encoded = []
    for g in df["gender"]:
        if g == "Male":
            gender_encoded.append(0)
        elif g == "Female":
            gender_encoded.append(1)
        else:
            raise ValueError("Unexpected value in gender column.")

    gender_encoded = pd.Series(gender_encoded, name="gender")

    # One-hot encoding education and occupation
    categorical_data = df[["education", "occupation"]]

    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_array = encoder.fit_transform(categorical_data)

    encoded_columns = encoder.get_feature_names_out(["education", "occupation"])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns)

    # Combining all features
    X = pd.concat([gender_encoded, encoded_df], axis=1)

    # Map the original income values to binary labels, for convenience
    income_encoded = []
    for val in df["income"]:
        if val == ">50K":
            income_encoded.append(1)
        elif val == "<=50K":
            income_encoded.append(0)
        elif val in [0, 1]: # Because of the simulation
            income_encoded.append(int(val))
        else:
            raise ValueError("Unexpected value in income column.")

    y = pd.Series(income_encoded, name="income")

    return X, y
