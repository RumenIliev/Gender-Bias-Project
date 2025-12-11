import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_dataframe(df):
    # Basic sanity checks
    required_columns = ["gender", "education", "occupation", "income"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()

    # Clean categorical variables in Adult Income dataset
    for col in ["gender", "education", "occupation"]:
        df[col] = df[col].astype(str).str.strip()

    df.replace("?", pd.NA, inplace=True) # Treat "?" as missing
    df = df.dropna(subset=required_columns)

    # Normalize category formats
    df["gender"] = df["gender"].str.capitalize()
    df["education"] = df["education"].str.lower()
    df["occupation"] = df["occupation"].str.lower()
    df = df[df["occupation"] != "armed-forces"] # Remove Armed Forces occupation

    # Map education to low, medium, high
    education_mapped = []

    for e in df["education"]:
        if e in ["preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "hs-grad"]:
            education_mapped.append("low")

        elif e in ["some-college", "assoc-acdm", "assoc-voc"]:
            education_mapped.append("medium")

        elif e in ["bachelors", "masters", "prof-school", "doctorate"]:
            education_mapped.append("high")

        elif e in ["low", "medium", "high"]: # because of the simulation
            education_mapped.append(e)
        else:
            raise ValueError("Unexpected education value.")

    df["education_mapped"] = education_mapped


    # Map occupation to manual, service, professional
    occupation_mapped = []

    for o in df["occupation"]:
        if o in ["craft-repair", "handlers-cleaners", "machine-op-inspct", "transport-moving", "farming-fishing"]:
            occupation_mapped.append("manual")

        elif o in ["other-service", "protective-serv", "priv-house-serv", "tech-support"]:
            occupation_mapped.append("service")

        elif o in ["exec-managerial", "prof-specialty", "sales", "adm-clerical"]:
            occupation_mapped.append("professional")

        elif o in ["manual", "service", "professional"]: # because of the simulation
            occupation_mapped.append(o)
        else:
            raise ValueError("Unexpected occupation value.")

    df["occupation_mapped"] = occupation_mapped

    # Encoding gender
    gender_encoded = []
    for g in df["gender"]:
        if g == "Male":
            gender_encoded.append(0)
        elif g == "Female":
            gender_encoded.append(1)
        else:
            raise ValueError("Unexpected value in gender column.")

    gender_encoded = pd.Series(gender_encoded, index=df.index, name="gender")


    # One-hot encoding education and occupation
    categorical_data = df[["education_mapped", "occupation_mapped"]]

    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_array = encoder.fit_transform(categorical_data)

    encoded_columns = encoder.get_feature_names_out(["education_mapped", "occupation_mapped"])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)

    # Combining all features
    X = pd.concat([gender_encoded, encoded_df], axis=1)

    # Map the original income values to binary labels, for convenience
    income_encoded = []
    for val in df["income"]:
        if isinstance(val, str):
            val = val.strip()
            if val == ">50K":
                income_encoded.append(1)
            elif val == "<=50K":
                income_encoded.append(0)
            else:
                raise ValueError("Unexpected label in income column.")
        elif val in [0, 1]: # because of the simulation
            income_encoded.append(int(val))
        else:
            raise ValueError("Unexpected value in income column.")

    y = pd.Series(income_encoded, index=df.index, name="income")

    return X, y
