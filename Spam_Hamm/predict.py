import pickle
import numpy as np
import pandas as pd

from naive_bayes import MultinomialNaiveBayesScratch


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

print("=" * 65)

print(
    "LOADING SAVED NAIVE BAYES MODEL"
)

print("=" * 65)


with open(
    "naive_bayes_model.pkl",
    "rb"
) as f:

    model_data = pickle.load(f)


# Extract model
model = model_data["model"]


# Extract feature names
feature_names = model_data[
    "feature_names"
]


print(
    "\nModel loaded successfully!"
)


print(
    f"Number of features expected: "
    f"{len(feature_names)}"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print(
    "\nLoading emails.csv..."
)


df = pd.read_csv(
    "emails.csv"
)


print(
    f"Dataset shape: {df.shape}"
)


# Save email IDs
email_ids = df.iloc[
    :, 0
].to_numpy()


# ============================================================
# 3. REMOVE EMAIL ID
# ============================================================

df_features = df.drop(
    columns=[df.columns[0]]
)


# ============================================================
# 4. REMOVE LABEL
# ============================================================

if "Prediction" in df_features.columns:

    X_new = df_features.drop(
        columns=["Prediction"]
    )

else:

    X_new = df_features


# ============================================================
# 5. CHECK FEATURES
# ============================================================

X_new = X_new.to_numpy(
    dtype=np.float64
)


expected_features = (
    len(feature_names)
)


actual_features = (
    X_new.shape[1]
)


print(
    f"\nExpected features: "
    f"{expected_features}"
)


print(
    f"Actual features:   "
    f"{actual_features}"
)


if expected_features != actual_features:

    raise ValueError(
        "Feature mismatch! "
        f"Expected {expected_features}, "
        f"but got {actual_features}."
    )


# ============================================================
# 6. PREDICT
# ============================================================

print(
    "\nMaking predictions..."
)


predictions = model.predict(
    X_new
)


# ============================================================
# 7. PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_new
)


# Find spam class index
spam_index = list(
    model.classes_
).index(1)


spam_probabilities = probabilities[
    :, spam_index
]


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "PREDICTION RESULTS"
)

print(
    "=" * 65
)


for i in range(
    len(predictions)
):

    if predictions[i] == 1:

        label = "SPAM"

    else:

        label = "NOT SPAM"


    print(
        f"{email_ids[i]:15} | "
        f"{label:10} | "
        f"Spam Probability: "
        f"{spam_probabilities[i]:.4f}"
    )


# ============================================================
# 9. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Email No.": email_ids,

    "Predicted": predictions,

    "Spam Probability":
        spam_probabilities

})


prediction_df.to_csv(
    "predictions.csv",
    index=False
)


print(
    "\nPredictions saved to:"
    " predictions.csv"
)


# ============================================================
# 10. SUMMARY
# ============================================================

spam_count = np.sum(
    predictions == 1
)


not_spam_count = np.sum(
    predictions == 0
)


print(
    "\n" + "=" * 65
)

print(
    "SUMMARY"
)

print(
    "=" * 65
)


print(
    f"Total emails : "
    f"{len(predictions)}"
)


print(
    f"Spam         : "
    f"{spam_count}"
)


print(
    f"Not Spam     : "
    f"{not_spam_count}"
)


print(
    "=" * 65
)