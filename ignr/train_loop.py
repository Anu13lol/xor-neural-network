import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

#What if ReLU:
def relu(z):
    return np.maximum(0, z)


np.random.seed(42) #Safe seeding

#XOR dataset:
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

#layer size:
input_n = 2
hidden_n = 2
output_n = 1

#Random weight init(small val):
W1 = np.random.randn(input_n, hidden_n) * 0.1
b1 = np.zeros((1, hidden_n))
W2 = np.random.randn(hidden_n, output_n) * 0.1
b2 = np.zeros((1,output_n))

#Forward pass:
def forward(X):
    #Layer 1
    z1 = X @ W1 + b1
    a1 = sigmoid(z1) #activation
    #a1 = relu(z1)

    #Layer 2
    z2 = a1 @ W2 + b2 #forwards the last activation
    a2 = sigmoid(z2) #final prediction

    cache = (z1, a1, z2, a2) #backpropogation datas
    return a2, cache #returning backpropogation datas and the final prediction

predictions, cache = forward(X)
print("predictions:\n", predictions)


lr = 0.5 #from forward pass testing

#Backpropogation:
def backward(X, Y, cache):
    global W1, b1, W2, b2
    z1, a1, z2, a2 = cache
    n = X.shape[0] #number of samples

    #Output layer gradients
    dl_da2 = 2 * (a2 - Y) / n
    dl_dz2 = dl_da2 * sigmoid_deriv(z2)
    dl_dW2 = a1.T @ dl_dz2
    dl_db2 = np.sum(dl_dz2, axis=0, keepdims=True)

    #Hidden layer gradients
    dl_da1 = dl_dz2 @ W2.T
    dl_dz1 = dl_da1 * sigmoid_deriv(z1)
    dl_dW1 = X.T @ dl_dz1
    dl_db1 = np.sum(dl_dz1, axis=0, keepdims=True)

    #Update weights and biases
    W1 -= lr * dl_dW1
    b1 -= lr * dl_db1
    W2 -= lr * dl_dW2
    b2 -= lr * dl_db2

for epoch in range(10000):
    predictions, cache = forward(X)
    loss = np.mean((predictions - Y) ** 2)
    backward(X, Y, cache)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

predictions, _ = forward(X)
print("\nFinal predictions:")
for i, (x, pred, target) in enumerate(zip(X, predictions, Y)):
    print(f"  {x} → {pred[0]:.3f}  (target: {target[0]})")




