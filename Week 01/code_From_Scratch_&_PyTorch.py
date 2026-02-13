# Using Numpy Alone

import numpy as np
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
np.random.seed(0)

X = np.random.randn(100, 2)                               # Datasets (100 samples, 2 features)
y = (X[:, 0] + X[:, 1] > 0).reshape(-1, 1).astype(float) 

input_size = 2                                            # Architecture of Model should be manually designed
hidden_size = 5
output_size = 1
lr = 0.1                                                  # The number of weight matrices = Number of Hidden Layers + 1.

W1 = np.random.randn(input_size, hidden_size)             # (Layer 1) Input_size == Number_of_Features.....Weights are iniatialized randomly to avoid symmetry
b1 = np.zeros((1, hidden_size))                           # Broadcasting (Bias) --> Automatic Adjust the Dimensions of Biases

W2 = np.random.randn(hidden_size, output_size)            
b2 = np.zeros((1, output_size))

for epoch in range(1000):

    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    loss = -np.mean(y * np.log(y_hat + 1e-8) +                  # 1e-8 is used to prevent log(0) explosion 
                     (1 - y) * np.log(1 - y_hat + 1e-8))        # But professional way,Clip y_hat to be between 1e-8 and 0.99999999, 
                                                                # y_hat_clipped = np.clip(y_hat, 1e-8, 1 - 1e-8)                                                     
                                                                # loss = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped)
    dz2 = y_hat - y
    dW2 = a1.T @ dz2 / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


val1 = float(input("Enter first feature: "))
val2 = float(input("Enter second feature: "))

user_input = np.array([[val1, val2]])

layer1_raw = user_input @ W1 + b1
layer1_act = np.maximum(0, layer1_raw)  
layer2_raw = layer1_act @ W2 + b2
prediction = 1 / (1 + np.exp(-layer2_raw)) 

final_choice = 1 if prediction > 0.5 else 0

print(f"Model Probability: {prediction[0,0]:.4f}")
print(f"Model Prediction: {final_choice}")

# Using PyTorch

import torch
import torch.nn as nn

torch.manual_seed(0)


model = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)


X = torch.randn(10, 2)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)


criterion = nn.BCELoss()


y_hat = model(X)
loss = criterion(y_hat, y)


model.zero_grad()    
loss.backward()       

