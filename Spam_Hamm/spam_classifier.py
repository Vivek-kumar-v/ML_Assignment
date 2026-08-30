
import numpy as np
import pandas as pd


def load_data(path="emails.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=[df.columns[0]])

    label_col = df.columns[-1]

    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float64)  
    y = df[label_col].to_numpy(dtype=np.int64)                   
    feature_names = df.drop(columns=[label_col]).columns.to_list()

    return X, y, feature_names


def train_test_split_manual(X, y, n_train=4500, seed=42):
    n_total = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)          

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_test, y_test


class MultinomialNaiveBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha              
        self.classes_ = None
        self.log_prior_ = None          
        self.log_likelihood_ = None     

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.log_prior_ = np.zeros(n_classes)
        self.log_likelihood_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]                     

            self.log_prior_[idx] = np.log(X_c.shape[0] / n_samples)

            word_counts_c = X_c.sum(axis=0)          
            total_words_c = word_counts_c.sum()      

            self.log_likelihood_[idx, :] = np.log(
                (word_counts_c + self.alpha) /
                (total_words_c + self.alpha * n_features)
            )

        return self

    def _joint_log_likelihood(self, X):
        return X @ self.log_likelihood_.T + self.log_prior_

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        """Optional: convert joint log-likelihoods to normalized probabilities."""
        jll = self._joint_log_likelihood(X)
        max_jll = np.max(jll, axis=1, keepdims=True)
        log_sum = max_jll + np.log(np.sum(np.exp(jll - max_jll), axis=1, keepdims=True))
        log_proba = jll - log_sum
        return np.exp(log_proba)


def confusion_counts(y_true, y_pred, positive_class=1):
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    tn = np.sum((y_pred != positive_class) & (y_true != positive_class))
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
    fn = np.sum((y_pred != positive_class) & (y_true == positive_class))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred, positive_class=1):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred, positive_class)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
    }


def main():
    print("Loading data...")
    X, y, feature_names = load_data("emails.csv")
    print(f"Full dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution -> Spam(1): {np.sum(y==1)}, Not-Spam(0): {np.sum(y==0)}")

    print("\nSplitting into train (4500) / test (672)...")
    X_train, y_train, X_test, y_test = train_test_split_manual(X, y, n_train=4500, seed=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("\nTraining Multinomial Naive Bayes (from scratch, Laplace smoothing, log-space)...")
    model = MultinomialNaiveBayesScratch(alpha=1.0)
    model.fit(X_train, y_train)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(y_test, y_pred, positive_class=1)

    print("\n" + "=" * 50)
    print("RESULTS ON TEST SET (672 emails)")
    print("=" * 50)
    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix -> TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print("=" * 50)

    y_train_pred = model.predict(X_train)
    train_metrics = compute_metrics(y_train, y_train_pred, positive_class=1)
    print("\n(For reference) Training set performance:")
    print(f"Accuracy  : {train_metrics['accuracy']:.4f}")
    print(f"Precision : {train_metrics['precision']:.4f}")
    print(f"Recall    : {train_metrics['recall']:.4f}")
    print(f"F1 Score  : {train_metrics['f1_score']:.4f}")

    return model, metrics, train_metrics


if __name__ == "__main__":
    main()