import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import json

class NeuralNetwork:
    """
    A simple feedforward neural network with backpropagation learning.
    Demonstrates the significance of backpropagation in training neural networks.
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases randomly
        self.weights = []
        self.biases = []
        
        # Initialize weights with small random values
        for i in range(self.num_layers - 1):
            weight_matrix = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.5
            bias_vector = np.random.randn(layer_sizes[i+1], 1) * 0.5
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
        
        # Track training history for visualization
        self.loss_history = []
        self.weight_changes = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward_pass(self, x):
        """
        Perform forward pass through the network.
        
        Args:
            x: Input data (column vector)
            
        Returns:
            Tuple of (activations, z_values) for each layer
        """
        activations = [x]
        z_values = []
        
        current_activation = x
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], current_activation) + self.biases[i]
            z_values.append(z)
            current_activation = self.sigmoid(z)
            activations.append(current_activation)
        
        return activations, z_values
    
    def compute_loss(self, predictions, targets):
        """Compute mean squared error loss"""
        return np.mean((predictions - targets) ** 2)
    
    def backpropagation(self, x, y):
        """
        Perform backpropagation to compute gradients.
        
        Args:
            x: Input data
            y: Target output
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        # Forward pass
        activations, z_values = self.forward_pass(x)
        
        # Initialize gradient lists
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        
        # Compute error for output layer
        delta = (activations[-1] - y) * self.sigmoid_derivative(z_values[-1])
        
        # Backpropagate the error
        for layer in range(self.num_layers - 2, -1, -1):
            # Compute gradients for current layer
            weight_gradients[layer] = np.dot(delta, activations[layer].T)
            bias_gradients[layer] = delta
            
            # Compute error for previous layer (if not input layer)
            if layer > 0:
                delta = np.dot(self.weights[layer].T, delta) * self.sigmoid_derivative(z_values[layer-1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradients"""
        weight_changes_epoch = []
        
        for i in range(len(self.weights)):
            # Store weight changes for analysis
            weight_change = self.learning_rate * weight_gradients[i]
            weight_changes_epoch.append(np.mean(np.abs(weight_change)))
            
            # Update parameters
            self.weights[i] -= weight_change
            self.biases[i] -= self.learning_rate * bias_gradients[i]
        
        self.weight_changes.append(weight_changes_epoch)
    
    def train(self, X, Y, epochs: int = 1000, verbose: bool = True):
        """
        Train the neural network using backpropagation.
        
        Args:
            X: Training inputs (each column is a sample)
            Y: Training targets (each column is a sample)
            epochs: Number of training epochs
            verbose: Whether to print training progress
        """
        for epoch in range(epochs):
            total_loss = 0
            
            # Train on each sample
            for i in range(X.shape[1]):
                x = X[:, i:i+1]
                y = Y[:, i:i+1]
                
                # Forward pass
                activations, _ = self.forward_pass(x)
                prediction = activations[-1]
                
                # Compute loss
                loss = self.compute_loss(prediction, y)
                total_loss += loss
                
                # Backpropagation
                weight_gradients, bias_gradients = self.backpropagation(x, y)
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            # Record average loss for this epoch
            avg_loss = total_loss / X.shape[1]
            self.loss_history.append(avg_loss)
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")
    
    def predict(self, x):
        """Make a prediction for given input"""
        activations, _ = self.forward_pass(x)
        return activations[-1]
    
    def evaluate_accuracy(self, X, Y, threshold: float = 0.5):
        """Evaluate classification accuracy"""
        correct = 0
        total = X.shape[1]
        
        for i in range(total):
            x = X[:, i:i+1]
            y = Y[:, i:i+1]
            prediction = self.predict(x)
            
            # Convert to binary classification
            pred_binary = (prediction > threshold).astype(int)
            actual_binary = (y > threshold).astype(int)
            
            if np.array_equal(pred_binary, actual_binary):
                correct += 1
        
        return correct / total
