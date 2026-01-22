# ML Algorithms from Scratch

A comprehensive collection of fundamental machine learning algorithms implemented from scratch using NumPy and Python. This repository serves as an educational resource for understanding the inner workings of core ML algorithms without relying on high-level libraries like scikit-learn or TensorFlow.

## 📚 Overview

This repository contains clean, well-documented implementations of essential machine learning algorithms, each designed to demonstrate the mathematical foundations and computational mechanics behind modern machine learning. All implementations prioritize clarity and educational value while maintaining computational efficiency.

## 🎯 Algorithms Included

### 1. Logistic Regression
**Location:** `logistic-regression/`

**What it is:** A linear classification algorithm that models the probability of a binary outcome using the logistic (sigmoid) function. It's essentially a single-layer perceptron that learns a linear decision boundary.

**Key Features:**
- Binary classification using sigmoid activation
- Gradient descent optimization with early stopping
- Configurable learning rate and tolerance
- Supports both NumPy arrays and Pandas DataFrames

**Intuition:** Imagine you're trying to separate two groups of points on a graph. Logistic regression finds the best line (or hyperplane in higher dimensions) that separates them, but instead of a hard boundary, it gives you probabilities. The sigmoid function squashes any real number into a value between 0 and 1, representing the probability of belonging to one class.

**Mathematical Foundation:** Uses maximum likelihood estimation to find weights that minimize the cross-entropy loss function. The gradient descent algorithm iteratively adjusts weights to reduce prediction error.

---

### 2. Decision Tree
**Location:** `decision-tree/`

**What it is:** A tree-based classifier that makes decisions by recursively splitting the data based on feature values. Each internal node represents a decision rule, and each leaf node represents a class prediction.

**Key Features:**
- Gini impurity for split selection
- Configurable maximum depth and feature sampling
- Recursive tree construction
- Handles both numerical and categorical features

**Intuition:** Think of a decision tree like a flowchart or a game of "20 Questions." Starting from the root, the algorithm asks a series of yes/no questions about the features (e.g., "Is age > 30?"). Based on the answer, it moves to the next question, eventually reaching a leaf node that gives the final prediction. The algorithm learns which questions to ask by finding splits that best separate different classes.

**Mathematical Foundation:** Uses Gini impurity to measure how "mixed" a node is. A node with all examples from one class has Gini = 0 (pure), while a node with equal distribution has Gini = 0.5 (impure). The algorithm greedily selects splits that minimize the weighted average Gini of child nodes.

---

### 3. Random Forest
**Location:** `random-forest/`

**What it is:** An ensemble method that combines multiple decision trees through majority voting. Each tree is trained on a random subset of data and features, making the model more robust and less prone to overfitting.

**Key Features:**
- Bootstrap aggregation (bagging) for diversity
- Configurable number of trees and subsample size
- Feature randomization at each split
- Majority voting for final predictions

**Intuition:** The wisdom of crowds applied to machine learning. Instead of relying on one decision tree (which might overfit), we train many trees on different random samples of data. Each tree "votes" on the final prediction, and we take the majority vote. This approach reduces variance and improves generalization. It's like asking multiple experts for their opinion and going with the consensus.

**Mathematical Foundation:** Combines the bias-variance tradeoff principle with bootstrap sampling. By training on different subsets, each tree captures different patterns in the data. The ensemble's prediction has lower variance than individual trees while maintaining low bias.

---

### 4. Multi-Layer Perceptron (2-Layer)
**Location:** `multi-layer-perceptron/neural_network_2_layers.py`

**What it is:** A simple neural network with one hidden layer that can learn non-linear decision boundaries. It's the foundation of deep learning, demonstrating how multiple layers of neurons can approximate complex functions.

**Key Features:**
- Single hidden layer with tanh activation
- Output layer with sigmoid (binary) or softmax (multi-class)
- Momentum-based gradient descent
- Mini-batch training support
- Automatic activation function selection based on output size

**Intuition:** If logistic regression is a single decision maker, a 2-layer MLP is a team of decision makers. The input features are processed by hidden neurons (each applying a transformation), and their outputs are combined to make the final decision. The hidden layer allows the network to learn non-linear patterns that a single layer cannot capture. Think of it as having multiple specialists who each look at the problem from a different angle, then combine their insights.

**Mathematical Foundation:** Implements backpropagation to compute gradients through the network. The chain rule of calculus allows error signals to flow backward from output to input, enabling efficient weight updates. Momentum helps accelerate convergence in the right direction.

---

### 5. Multi-Layer Perceptron (N-Layer)
**Location:** `multi-layer-perceptron/neural_network_N_layers.py`

**What it is:** A flexible, deep neural network framework that supports arbitrary numbers of layers and multiple activation functions. This is a fully customizable neural network implementation.

**Key Features:**
- Configurable number of layers
- Multiple activation functions: tanh, ReLU, sigmoid, softmax
- L2 regularization support
- Early stopping with patience
- He initialization for better convergence
- Validation set monitoring

**Intuition:** This is the deep learning workhorse. By stacking multiple layers, the network can learn hierarchical representations: early layers detect simple patterns (edges, curves), middle layers combine them into complex features (shapes, objects), and later layers make high-level decisions. Each layer builds upon the abstractions learned by previous layers, enabling the network to model extremely complex relationships.

**Mathematical Foundation:** Extends backpropagation to arbitrary depth. Uses He initialization (scaled by √(2/n)) for ReLU activations to prevent vanishing gradients. L2 regularization adds a penalty term to the loss function, encouraging smaller weights and reducing overfitting.

---

### 6. Hidden Markov Model (HMM)
**Location:** `hidden-markov-model/`

**What it is:** A probabilistic model for sequence labeling tasks, particularly useful for Named Entity Recognition (NER). It models sequences where each observation depends on a hidden state, and states transition according to Markov assumptions.

**Key Features:**
- Sequence labeling with BIO tagging scheme
- Transition and emission probability matrices
- Viterbi-like decoding for prediction
- Text preprocessing with lemmatization

**Intuition:** Imagine you're trying to identify names in a sentence, but you can only see the words (observations), not the labels (hidden states). An HMM assumes that: (1) the current label depends only on the previous label (Markov property), and (2) each word's probability depends on its label. The model learns patterns like "after a 'B' (beginning of name), we often see 'I' (inside name)" and "the word 'Alvin' is more likely to be a name than a verb."

**Mathematical Foundation:** Uses the forward-backward algorithm principles. The emission probabilities (P(word|label)) and transition probabilities (P(current_label|previous_label)) are learned from training data. Prediction uses maximum likelihood decoding to find the most probable sequence of hidden states.

---

## 📁 Repository Structure

```
ML_Algorithms/
│
├── logistic-regression/
│   ├── logistic_regression.py      # Core implementation
│   └── logistic_regression.ipynb    # Example usage and evaluation
│
├── decision-tree/
│   └── decision_tree.py             # Decision tree classifier
│
├── random-forest/
│   └── random_forest.py             # Random forest ensemble
│
├── multi-layer-perceptron/
│   ├── neural_network_2_layers.py  # 2-layer MLP implementation
│   ├── neural_network_N_layers.py  # N-layer MLP framework
│   ├── neural_networks_2_layers.ipynb
│   └── neural_networks_N_layers.ipynb
│
├── hidden-markov-model/
│   └── hidden_markov_model.ipynb   # HMM for NER task
│
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

For Hidden Markov Model:
```bash
pip install nltk
```

### Basic Usage Examples

#### Logistic Regression
```python
from logistic_regression.logistic_regression import Logistic_Regression

# Initialize model
model = Logistic_Regression(iterations=10000, learning_rate=0.00001)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

#### Decision Tree
```python
from decision_tree.decision_tree import DecisionTree

# Initialize with max depth
model = DecisionTree(max_depth=10, random_state=42)

# Train and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### Random Forest
```python
from random_forest.random_forest import RandomForest

# Initialize with 100 trees
model = RandomForest(num_trees=100, max_depth=10, random_state=42)

# Train and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### Multi-Layer Perceptron (N-Layer)
```python
from neural_network_N_layers import Model, Layer

# Define architecture
layers = [
    Layer(units=128, input_units=784, activation='relu'),
    Layer(units=64, activation='relu'),
    Layer(units=10, activation='softmax')
]

# Create and train model
model = Model(layers, random_state=42)
model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          epochs=100,
          learning_rate=0.001,
          patience=10)
```

## 🎓 Educational Value

This repository is designed for:
- **Students** learning machine learning fundamentals
- **Practitioners** wanting to understand algorithm internals
- **Researchers** needing baseline implementations
- **Anyone** curious about how ML algorithms actually work

Each implementation includes:
- Clear mathematical foundations
- Detailed code comments
- Intuitive explanations
- Practical usage examples

## 🔬 Key Learning Outcomes

By studying these implementations, you'll understand:
- How gradient descent optimizes model parameters
- The mechanics of backpropagation in neural networks
- How decision trees select optimal splits
- Ensemble methods and their benefits
- Probabilistic models for sequence data
- The importance of regularization and early stopping

## 📝 Notes

- All implementations use NumPy for numerical computations
- Code is optimized for clarity over performance (though still efficient)
- Jupyter notebooks provide interactive examples and visualizations
- Some algorithms include comparisons with scikit-learn implementations

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new algorithms
- Improve existing implementations
- Fix bugs or optimize code
- Enhance documentation
- Add more examples

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Tan Pengshi Alvin**

---

*"Understanding the fundamentals is the key to mastering any field. These implementations strip away the abstractions to reveal the beautiful mathematics and algorithms that power modern machine learning."*
