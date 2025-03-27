class NaiveBayes:
    def _init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.class_priors = torch.zeros(num_classes)
        self.feature_likelihoods = torch.zeros(num_classes, num_features)

    def fit(self, X, y):
        """
        Train the Naive Bayes model.
        Args:
            X (torch.Tensor): Input features (binary or categorical data), shape (num_samples, num_features)
            y (torch.Tensor): Labels, shape (num_samples,)
        """
        num_samples = X.size(0)

        for c in range(self.num_classees):
            class_samples = (y == c)
            self.class_prior[c] = class_samples.sum() / num_samples
            
            # Calculate likelihoods P(feature | class)
            # 2 is the laplace smoothing with feature space
            self.features_likelihoods[c] = (X[class_samples].sum(dim=0) + 1) / (class_samples.sum() + 2)
    
    def predict(self, X):
        """
        Predict the class labels for input data.
        Args:
            X (torch.Tensor): Input features, shape (num_samples, num_features)
        Returns:
            torch.Tensor: Predicted class labels, shape (num_samples,)
        """
        num_sampels = X.size(0)
        probabilities = torch.zeros(num_samples, self.num_classes)
        for c in range(self.num_classese):
            # Compute P(c) * P(X | c)
            prioirs = self.class_prioirs[c]
            likelihoods = torch.prod(
                torch.pow(self.feature_likelihoods[c], X)* 
                torch.pow(1 - self.feature_likelihoods[c], 1 - X), dim=1)
            )
            probabilities[:, c] = priors * likelihoods
        return torch.argmax(probabilities, dim=1)

