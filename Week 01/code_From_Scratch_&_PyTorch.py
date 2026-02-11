
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

    # ----- Forward pass -----
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    # ----- Loss (Binary Cross-Entropy) -----
    loss = -np.mean(y * np.log(y_hat + 1e-8) +
                     (1 - y) * np.log(1 - y_hat + 1e-8))

    # ----- Backpropagation -----
    dz2 = y_hat - y
    dW2 = a1.T @ dz2 / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # ----- Update -----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
# --- PREDICTION FOR A NEW USER ---

# 1. Get input from the user
val1 = float(input("Enter first feature: "))
val2 = float(input("Enter second feature: "))

# 2. Put it in the right shape (1 row, 2 columns)
# This matches the shape the model expects
user_input = np.array([[val1, val2]])

# 3. The Forward Pass (Just the math, no learning here)
layer1_raw = user_input @ W1 + b1
layer1_act = np.maximum(0, layer1_raw)  # ReLU
layer2_raw = layer1_act @ W2 + b2
prediction = 1 / (1 + np.exp(-layer2_raw)) # Sigmoid

# 4. Turn the decimal into a 0 or 1
final_choice = 1 if prediction > 0.5 else 0

print(f"Model Probability: {prediction[0,0]:.4f}")
print(f"Model Prediction: {final_choice}")
