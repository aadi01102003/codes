class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, task='classification'):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.task = task  # 'classification' or 'regression'
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        
        if self.task == 'classification':
            self.output = self.sigmoid(self.output_input)
        elif self.task == 'regression':
            self.output = self.output_input  # Linear activation for regression
            
        return self.output
    
    def backward(self, X, y, output):
        output_error = y - output
        
        if self.task == 'classification':
            output_delta = output_error * self.sigmoid_derivative(output)
        elif self.task == 'regression':
            output_delta = output_error  # Derivative of linear activation is 1
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        output = self.forward(X)
        if self.task == 'classification':
            return (output > 0.5).astype(int)  # Convert probabilities to 0 or 1
        return output  # For regression, return raw output
# XOR Logic Gate Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train the network for classification
nn_classification = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, task='classification')
nn_classification.train(X, y, epochs=10000)

# Predict and print results
print("Classification Predictions:")
print(nn_classification.predict(X))
