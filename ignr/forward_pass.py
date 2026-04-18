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

def forward(X, W1, b1, W2, b2):
    #Layer 1
    z1 = X @ W1 + b1
    a1 = sigmoid(z1) #activation
    #a1 = relu(z1)

    #Layer 2
    z2 = a1 @ W2 + b2 #forwards the last activation
    a2 = sigmoid(z2) #final prediction

    cache = (z1, a1, z2, a2) #backpropogation datas
    return a2, cache #returning backpropogation datas and the final prediction

#Executing the forward pass with our initial datas
predictions, cache = forward(X, W1, b1, W2, b2)
print("predictions:\n", predictions)



#For Loss calculations:
def mse(predictions, targets):
    return np.mean((predictions - targets)**2)

loss = mse(predictions, Y)
print(f"Loss: {loss:.4f}")



