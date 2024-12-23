class SVM:
    def __init__(self, lr, epochs, c):
        self.lr = lr
        self.epoch = epoch
        self.c = c
        self.w = None
        self.b = None
        

    def fit(self, x, y):
        n_samples, n__features = x.shape
        limit = 1/np.sqrt(n_samples)
        self.w = np.random.uniform(-limit, limit, n_features)
        self.b = 0

        for _ in range(len(self.epochs)):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for idx in indices:
                xi = x[idx]
                yi = y[idx]
                condition = y * (xi @ w.T + self.b)
                if condition < 1:
                    dw = -y * x
                    db = -y
                else:
                    dw = self.w
                    db = 0
                self.w -= self.lr * dw
                self.b -= self.lr * db 

    
    def predict(self, x):
        o = (x @ self.w + self.b > 0)
        return np.sign(o).astype(int)
