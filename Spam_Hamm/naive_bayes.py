import numpy as np


class MultinomialNaiveBayesScratch:

    def __init__(self, alpha=1.0):
        # Laplace smoothing parameter
        self.alpha = alpha

        # Classes: [0, 1]
        self.classes_ = None

        # log P(class)
        self.log_prior_ = None

        # log P(word | class)
        self.log_likelihood_ = None


    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------
    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Find unique classes
        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)

        # Store log priors
        self.log_prior_ = np.zeros(n_classes)

        # Store log likelihoods
        self.log_likelihood_ = np.zeros(
            (n_classes, n_features)
        )


        # -----------------------------------------------------
        # Calculate probabilities for every class
        # -----------------------------------------------------

        for idx, c in enumerate(self.classes_):

            # Select all emails belonging to class c
            X_c = X[y == c]


            # -------------------------------------------------
            # Prior probability
            #
            # P(c) = number of documents in class c
            #        -------------------------------
            #             total documents
            # -------------------------------------------------

            prior = X_c.shape[0] / n_samples

            self.log_prior_[idx] = np.log(prior)


            # -------------------------------------------------
            # Word counts
            # -------------------------------------------------

            # Total occurrence of every word
            word_counts = X_c.sum(axis=0)

            # Total number of words in this class
            total_words = word_counts.sum()


            # -------------------------------------------------
            # Laplace smoothing
            #
            # P(word | class) =
            #
            # count(word,class) + alpha
            # --------------------------
            # total_words + alpha * V
            # -------------------------------------------------

            probabilities = (
                word_counts + self.alpha
            ) / (
                total_words +
                self.alpha * n_features
            )


            # Store log probabilities
            self.log_likelihood_[idx] = np.log(
                probabilities
            )


        return self


    # ---------------------------------------------------------
    # Calculate joint log likelihood
    # ---------------------------------------------------------
    def _joint_log_likelihood(self, X):

        return (
            X @ self.log_likelihood_.T
            + self.log_prior_
        )


    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
    def predict(self, X):

        jll = self._joint_log_likelihood(X)

        # Select class with maximum probability
        predictions = self.classes_[
            np.argmax(jll, axis=1)
        ]

        return predictions


    # ---------------------------------------------------------
    # PREDICT PROBABILITY
    # ---------------------------------------------------------
    def predict_proba(self, X):

        jll = self._joint_log_likelihood(X)


        # Log-sum-exp trick
        max_jll = np.max(
            jll,
            axis=1,
            keepdims=True
        )


        log_sum = (
            max_jll
            + np.log(
                np.sum(
                    np.exp(jll - max_jll),
                    axis=1,
                    keepdims=True
                )
            )
        )


        log_proba = jll - log_sum


        return np.exp(log_proba)