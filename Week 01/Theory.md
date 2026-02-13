# Perceptron - Neural Network Fundamentals
## Overview
A Perceptron is the fundamental building block of artificial neural networks. Introduced by Frank Rosenblatt in 1957, it represents the simplest form of a neural unit that makes binary decisions by combining inputs with learned weights and applying an activation function. Perceptrons are primarily used for binary classification tasks and form the foundational architecture for more complex deep learning models.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/6e3ee4de-a45f-4259-a979-73a554fa4609" />


## Key Characteristics
- Binary Decision Making: Produces outputs of either 0 or 1
- Linear Classifier: Creates linear decision boundaries
- Supervised Learning: Learns from labeled training data
- Foundation Architecture: Basis for multi-layer neural networks

## Architecture Components
### 1. Inputs `(x₁, x₂, ..., xₙ)`
- Features or measurable attributes that provide signals for decision making.
Example: For an OR gate implementation:
Inputs: (x₁, x₂) ∈ {0, 1}²

### 2. Weights `(w₁, w₂, ..., wₙ)`
- Parameters that determine each input's influence on the output. Learned during training through error correction.
- Properties:
- Represent feature importance
- Higher absolute values indicate stronger influence
- Initialized randomly, then optimized

### 3. Bias (b)
- Constant term that shifts the decision boundary independently of inputs.
- Purpose:
- Allows classification when all inputs are zero
- Prevents forcing the boundary through origin
- Controls activation threshold
  
### 4. Weighted Sum (Net Input)
Linear combination of inputs and weights with bias:
`z = Σ(wᵢ * xᵢ) + b   for i = 1 to n`

### 5. Activation Function
Typically uses a step function to produce binary output:
`ŷ = { 1 if z ≥ 0
      0 otherwise }`
# Mathematical Model
## Forward Propagation
Given: Input vector x, weight vector w, bias b
1. Compute weighted sum: `z = w·x + b`
2. Apply activation: `ŷ = step(z)`
Decision Boundary
The perceptron defines a hyperplane:
`w·x + b = 0`
All points where `w·x + b ≥ 0` are classified as 1, others as 0.

## Training Algorithm
Perceptron Learning Rule
Iteratively updates weights based on misclassification errors.

## Algorithm Steps:
- Initialize weights and bias (typically to zeros or small random values)
- For each training sample (x, y):
- Compute prediction: `ŷ = step(w·x + b)`
- Calculate error: `error = y - ŷ`
- Update weights: `wᵢ ← wᵢ + η * error * xᵢ`
- Update bias: `b ← b + η * error`
Repeat for multiple epochs until convergence
Where:
η = Learning rate `(0 < η ≤ 1),`
y = True label `(0 or 1),`
ŷ = Predicted label

## Convergence Theorem
For linearly separable data, the perceptron algorithm is guaranteed to converge to a solution in finite steps.

## Perceptron vs. Neural Networks
1. Single Perceptron Limitations
2. Only linear decision boundaries
3. Cannot solve XOR problem
4. Limited to binary classification

## Multi-Layer Extension
1. Neural networks overcome limitations by:
2. Multiple Layers: Input, hidden, and output layers
3. Non-linear Activations: Sigmoid, ReLU, Tanh
4. Backpropagation: Efficient gradient-based learning

## Neural Network Layer Computation:
Hidden layer: `z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾, a⁽¹⁾ = σ(z⁽¹⁾)`
Output layer: `z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾, ŷ = σ(z⁽²⁾)`

# Difference between Logistic Regression and Perceptron
Same Architecture, Different "Brain"
Both have identical structure:
Inputs → Weights → Summation `(z = w·x + b)` → Activation → Output
### 1. Activation Function
Perceptron---------Logistic Regression

<br>

Step Function------	Sigmoid Function

<br>

Output: 0 or 1------	Output: 0 to 1 (probability)

### 2. Learning Method
#### Perceptron
- Perceptron Learning Rule
- Updates only when wrong
- Stops once 100% correct
- No probability concept	
#### Logistic Regression
- Gradient Descent with Log-Loss
- Always updates (even when right)
- Keeps improving confidence
- Minimizes prediction uncertainty
### 3. Output Interpretation
**python**
Example: 
`z = 2.5`
- Perceptron:
`ŷ = 1 ` # "It's class 1"
- Logistic Regression:
`ŷ = 0.924`  # "92.4% probability it's class 1"
### 4. Convergence
- Perceptron	Logistic Regression
- ONLY if data is linearly separable	Always converges
- May not converge for noisy data	Handles overlapping classes
- Binary updates	Continuous optimization
### 5. Decision Boundary
Both create linear boundaries, but:
- Perceptron: Any line that separates classes
- Logistic Regression: "Best" line maximizing probability confidence
  
# MLP(Multi-Layer Perceptron)
Multi-Layer Perceptron (MLP) consists of fully connected dense layers that transform input data from one dimension to another. It is called multi-layer because it contains an input layer, one or more hidden layers and an output layer. The purpose of an MLP is to model complex relationships between inputs and outputs.

## Components of Multi-Layer Perceptron (MLP)

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/4607de94-0dae-4942-8fa3-ef3198d58c90" />

**Input Layer:** Each neuron or node in this layer corresponds to an input feature. For instance, if you have three input features the input layer will have three neurons.

**Hidden Layers:** MLP can have any number of hidden layers with each layer containing any number of nodes. These layers process the information received from the input layer.

**Output Layer:** The output layer generates the final prediction or result. If there are multiple outputs, the output layer will have a corresponding number of neurons.
Every connection in the diagram is a representation of the fully connected nature of an MLP. This means that every node in one layer connects to every node in the next layer. As the data moves through the network each layer transforms it until the final output is generated in the output layer.
Working of Multi-Layer Perceptron

## 1. Forward Propagation
In forward propagation the data flows from the input layer to the output layer, passing through any hidden layers. Each neuron in the hidden layers processes the input as follows:

1. **Weighted Sum**: The neuron computes the weighted sum of the inputs:
`z = ∑ᵢ wᵢxᵢ + b`

Where:
- `xᵢ` is the input feature.
- `wᵢ` is the corresponding weight.
- `b` is the bias term.

2. **Activation Function**: The weighted sum `z` is passed through an activation function to introduce non-linearity. Common activation functions include:
   - **Sigmoid**: `σ(z) = 1/(1 + e⁻ᶻ)`
   - **ReLU (Rectified Linear Unit)**: `f(z) = max(0, z)`
   - **Tanh (Hyperbolic Tangent)**: `tanh(z) = 2/(1 + e⁻²ᶻ) - 1`

## 2. Loss Function
Once the network generates an output, the next step is to calculate the loss using a loss function. In supervised learning, this compares the predicted output to the actual label.

For a classification problem, the commonly used binary cross-entropy loss function is:
`L = - (1/N) ∑ᵢ [yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ)]`

Where:
- `yᵢ` is the actual label
- `ŷᵢ` is the predicted label
- `N` is the number of samples

For regression problems, the mean squared error (MSE) is often used:
`MSE = (1/N) ∑ᵢ (yᵢ - ŷᵢ)²`

## 3. Backpropagation
The goal of training an MLP is to minimize the loss function by adjusting the network's weights and biases. This is achieved through backpropagation:

1. **Gradient Calculation**: The gradients of the loss function with respect to each weight and bias are calculated using the chain rule of calculus.
2. **Error Propagation**: The error is propagated back through the network, layer by layer.
3. **Gradient Descent**: The network updates the weights and biases by moving in the opposite direction of the gradient to reduce the loss:
`w = w - η ⋅ ∂L/∂w
b = b - η ⋅ ∂L/∂b`

Where:
- `w` is the weight
- `b` is the bias
- `η` is the learning rate
- `∂L/∂w` is the gradient of the loss function with respect to the weight
- `∂L/∂b` is the gradient of the loss function with respect to the bias

## 4. Optimization
MLPs rely on optimization algorithms to iteratively refine the weights and biases during training. Popular optimization methods include:

1. **Stochastic Gradient Descent (SGD)**: Updates the weights based on a single sample or a small batch of data:
`w = w - η ⋅ ∂L/∂w`,
`b = b - η ⋅ ∂L/∂b`

2. **Adam Optimizer**: An extension of SGD that incorporates momentum and adaptive learning rates for more efficient training:
`mₜ = β₁mₜ₋₁ + (1 - β₁) ⋅ gₜ,
vₜ = β₂vₜ₋₁ + (1 - β₂) ⋅ gₜ²`

Here `gₜ` represents the gradient at time `t`, and `β₁, β₂` are decay rates.






