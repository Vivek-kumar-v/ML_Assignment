import numpy as np
import pandas as pd
import pickle

from naive_bayes import MultinomialNaiveBayesScratch


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path="emails.csv"):

    df = pd.read_csv(path)

    print("\nDataset loaded successfully.")

    print(
        f"Original dataset shape: {df.shape}"
    )


    # --------------------------------------------------------
    # First column = Email No.
    # --------------------------------------------------------

    email_ids = df.iloc[:, 0].to_numpy()


    # --------------------------------------------------------
    # Remove Email No.
    # --------------------------------------------------------

    df = df.drop(
        columns=[df.columns[0]]
    )


    # --------------------------------------------------------
    # Last column = Prediction
    # --------------------------------------------------------

    label_col = df.columns[-1]


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df.drop(
        columns=[label_col]
    ).to_numpy(
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y = df[label_col].to_numpy(
        dtype=np.int64
    )


    # Feature names
    feature_names = (
        df.drop(
            columns=[label_col]
        ).columns.to_list()
    )


    return (
        X,
        y,
        email_ids,
        feature_names
    )


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

def train_test_split_manual(
    X,
    y,
    email_ids,
    n_train=4500,
    seed=42
):

    n_total = X.shape[0]


    # Random number generator
    rng = np.random.default_rng(seed)


    # Random permutation
    indices = rng.permutation(
        n_total
    )


    # First 4500 = training
    train_idx = indices[:n_train]


    # Remaining 672 = testing
    test_idx = indices[n_train:]


    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]


    email_ids_train = email_ids[
        train_idx
    ]

    email_ids_test = email_ids[
        test_idx
    ]


    return (
        X_train,
        y_train,
        X_test,
        y_test,
        email_ids_train,
        email_ids_test
    )


# ============================================================
# 3. CONFUSION MATRIX
# ============================================================

def confusion_counts(
    y_true,
    y_pred,
    positive_class=1
):

    tp = np.sum(
        (y_true == positive_class)
        &
        (y_pred == positive_class)
    )


    tn = np.sum(
        (y_true != positive_class)
        &
        (y_pred != positive_class)
    )


    fp = np.sum(
        (y_true != positive_class)
        &
        (y_pred == positive_class)
    )


    fn = np.sum(
        (y_true == positive_class)
        &
        (y_pred != positive_class)
    )


    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn)
    )


# ============================================================
# 4. METRICS
# ============================================================

def compute_metrics(
    y_true,
    y_pred
):

    tp, tn, fp, fn = confusion_counts(
        y_true,
        y_pred
    )


    # Accuracy
    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn)
    )


    # Precision
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    # Recall
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    # F1
    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


# ============================================================
# 5. SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_names,
    filename="naive_bayes_model.pkl"
):

    model_data = {

        "model": model,

        "feature_names":
            feature_names
    }


    with open(
        filename,
        "wb"
    ) as f:

        pickle.dump(
            model_data,
            f
        )


    print(
        f"\nModel saved successfully:"
        f" {filename}"
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "SPAM EMAIL CLASSIFICATION"
    )

    print(
        "Multinomial Naive Bayes "
        "from Scratch"
    )

    print("=" * 65)


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y, email_ids, feature_names = (
        load_data("emails.csv")
    )


    print(
        f"\nNumber of emails: "
        f"{X.shape[0]}"
    )

    print(
        f"Number of features: "
        f"{X.shape[1]}"
    )


    print(
        f"\nSpam emails: "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Not-spam emails: "
        f"{np.sum(y == 0)}"
    )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print(
        "\nCreating random train/test split..."
    )


    (
        X_train,
        y_train,
        X_test,
        y_test,
        email_ids_train,
        email_ids_test
    ) = train_test_split_manual(
        X,
        y,
        email_ids,
        n_train=4500,
        seed=42
    )


    print(
        f"Training samples: "
        f"{X_train.shape[0]}"
    )

    print(
        f"Testing samples: "
        f"{X_test.shape[0]}"
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining Naive Bayes..."
    )


    model = MultinomialNaiveBayesScratch(
        alpha=1.0
    )


    model.fit(
        X_train,
        y_train
    )


    print(
        "Training completed."
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        feature_names
    )


    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    print(
        "\nPredicting test emails..."
    )


    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = compute_metrics(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "FINAL RESULTS ON TEST SET"
    )

    print(
        "=" * 65
    )


    print(
        f"True Positive  : "
        f"{metrics['TP']}"
    )

    print(
        f"True Negative  : "
        f"{metrics['TN']}"
    )

    print(
        f"False Positive : "
        f"{metrics['FP']}"
    )

    print(
        f"False Negative : "
        f"{metrics['FN']}"
    )


    print(
        "\nAccuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        "Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        "Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        "F1 Score  : "
        f"{metrics['f1_score']:.4f}"
    )


    print(
        "=" * 65
    )


    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )


    # Find index corresponding to spam class
    spam_index = list(
        model.classes_
    ).index(1)


    spam_probabilities = probabilities[
        :, spam_index
    ]


    prediction_df = pd.DataFrame({

        "Email No.": email_ids_test,

        "Actual": y_test,

        "Predicted": y_pred,

        "Spam Probability":
            spam_probabilities

    })


    prediction_df.to_csv(
        "test_predictions.csv",
        index=False
    )


    print(
        "\nTest predictions saved to:"
        " test_predictions.csv"
    )


    # --------------------------------------------------------
    # Training performance
    # --------------------------------------------------------

    print(
        "\nCalculating training performance..."
    )


    y_train_pred = model.predict(
        X_train
    )


    train_metrics = compute_metrics(
        y_train,
        y_train_pred
    )


    print(
        "\nTraining Set Performance"
    )

    print(
        "-" * 40
    )

    print(
        f"Accuracy  : "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{train_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{train_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{train_metrics['f1_score']:.4f}"
    )


    print(
        "\nDone!" 
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()