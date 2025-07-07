import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import seaborn as sns

def create_xor_dataset():
    """
    Create XOR dataset - a classic non-linearly separable problem
    that demonstrates the need for hidden layers and backpropagation.
    """
    # XOR truth table
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)
    Y = np.array([[0, 1, 1, 0]], dtype=float)
    
    return X, Y

def create_complex_dataset(n_samples=200):
    """
    Create a more complex non-linear dataset for demonstration.
    """
    np.random.seed(42)
    
    # Create two concentric circles
    theta = np.random.uniform(0, 2*np.pi, n_samples//2)
    r1 = np.random.uniform(0.5, 1.0, n_samples//2)
    r2 = np.random.uniform(1.5, 2.0, n_samples//2)
    
    # Inner circle (class 0)
    x1_inner = r1 * np.cos(theta)
    x2_inner = r1 * np.sin(theta)
    y_inner = np.zeros(n_samples//2)
    
    # Outer circle (class 1)
    x1_outer = r2 * np.cos(theta)
    x2_outer = r2 * np.sin(theta)
    y_outer = np.ones(n_samples//2)
    
    # Combine datasets
    X = np.vstack([np.hstack([x1_inner, x1_outer]),
                   np.hstack([x2_inner, x2_outer])])
    Y = np.array([np.hstack([y_inner, y_outer])])
    
    return X, Y

def demonstrate_backpropagation_significance():
    """
    Demonstrate the significance of backpropagation by comparing:
    1. Network WITH backpropagation (learning)
    2. Network WITHOUT backpropagation (random weights)
    """
    print("=" * 60)
    print("BACKPROPAGATION SIGNIFICANCE DEMONSTRATION")
    print("=" * 60)
    
    # Create XOR dataset
    X_xor, Y_xor = create_xor_dataset()
    
    print("\n1. XOR Problem (Classic Non-Linear Problem)")
    print("-" * 40)
    print("XOR Truth Table:")
    print("Input: [0,0] -> Output: 0")
    print("Input: [0,1] -> Output: 1")
    print("Input: [1,0] -> Output: 1")
    print("Input: [1,1] -> Output: 0")
    
    # Network architecture: 2 inputs -> 4 hidden -> 1 output
    layer_sizes = [2, 4, 1]
    
    print(f"\nNeural Network Architecture: {layer_sizes}")
    print("Activation Function: Sigmoid")
    print("Learning Rate: 0.5")
    
    # Network WITH backpropagation
    print("\n" + "="*50)
    print("TRAINING WITH BACKPROPAGATION")
    print("="*50)
    
    nn_with_bp = NeuralNetwork(layer_sizes, learning_rate=0.5)
    nn_with_bp.train(X_xor, Y_xor, epochs=2000, verbose=True)
    
    # Network WITHOUT backpropagation (random weights)
    print("\n" + "="*50)
    print("NETWORK WITHOUT BACKPROPAGATION (Random Weights)")
    print("="*50)
    
    nn_without_bp = NeuralNetwork(layer_sizes, learning_rate=0.5)
    # Don't train - keep random weights
    
    # Evaluate both networks
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    
    print("\nTesting on XOR dataset:")
    print("\nWith Backpropagation:")
    for i in range(X_xor.shape[1]):
        x = X_xor[:, i:i+1]
        y = Y_xor[:, i:i+1]
        pred = nn_with_bp.predict(x)
        print(f"Input: {x.flatten()} -> Predicted: {pred[0,0]:.4f}, Actual: {y[0,0]}")
    
    print("\nWithout Backpropagation (Random Weights):")
    for i in range(X_xor.shape[1]):
        x = X_xor[:, i:i+1]
        y = Y_xor[:, i:i+1]
        pred = nn_without_bp.predict(x)
        print(f"Input: {x.flatten()} -> Predicted: {pred[0,0]:.4f}, Actual: {y[0,0]}")
    
    # Calculate accuracies
    acc_with_bp = nn_with_bp.evaluate_accuracy(X_xor, Y_xor)
    acc_without_bp = nn_without_bp.evaluate_accuracy(X_xor, Y_xor)
    
    print(f"\nAccuracy with Backpropagation: {acc_with_bp:.2%}")
    print(f"Accuracy without Backpropagation: {acc_without_bp:.2%}")
    
    return nn_with_bp, nn_without_bp

def visualize_learning_process(nn):
    """
    Visualize the learning process through loss curves and weight changes.
    """
    print("\n" + "="*50)
    print("LEARNING PROCESS VISUALIZATION")
    print("="*50)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backpropagation Learning Process Analysis', fontsize=16)
    
    # Plot 1: Loss over epochs
    axes[0, 0].plot(nn.loss_history, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Weight changes over epochs
    if nn.weight_changes:
        weight_changes_array = np.array(nn.weight_changes)
        for i in range(weight_changes_array.shape[1]):
            axes[0, 1].plot(weight_changes_array[:, i], label=f'Layer {i+1}')
        axes[0, 1].set_title('Weight Changes Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Average Weight Change')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weight distribution heatmap (first layer)
    im1 = axes[1, 0].imshow(nn.weights[0], cmap='RdBu', aspect='auto')
    axes[1, 0].set_title('Learned Weights (Input to Hidden Layer)')
    axes[1, 0].set_xlabel('Input Neurons')
    axes[1, 0].set_ylabel('Hidden Neurons')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Plot 4: Weight distribution heatmap (second layer)
    im2 = axes[1, 1].imshow(nn.weights[1], cmap='RdBu', aspect='auto')
    axes[1, 1].set_title('Learned Weights (Hidden to Output Layer)')
    axes[1, 1].set_xlabel('Hidden Neurons')
    axes[1, 1].set_ylabel('Output Neurons')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('backpropagation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_complex_problem():
    """
    Demonstrate backpropagation on a more complex dataset.
    """
    print("\n" + "="*60)
    print("COMPLEX PROBLEM DEMONSTRATION")
    print("="*60)
    
    # Create complex dataset
    X_complex, Y_complex = create_complex_dataset(400)
    
    print("Created a complex dataset with two concentric circles")
    print(f"Dataset shape: {X_complex.shape[1]} samples, {X_complex.shape[0]} features")
    
    # Create and train network
    nn_complex = NeuralNetwork([2, 8, 8, 1], learning_rate=0.1)
    print("\nTraining network on complex dataset...")
    nn_complex.train(X_complex, Y_complex, epochs=1000, verbose=True)
    
    # Evaluate accuracy
    accuracy = nn_complex.evaluate_accuracy(X_complex, Y_complex)
    print(f"\nFinal accuracy on complex dataset: {accuracy:.2%}")
    
    # Visualize the decision boundary
    visualize_decision_boundary(nn_complex, X_complex, Y_complex)
    
    return nn_complex

def visualize_decision_boundary(nn, X, Y):
    """
    Visualize the decision boundary learned by the neural network.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Data points
    plt.subplot(1, 2, 1)
    colors = ['red' if y == 0 else 'blue' for y in Y[0]]
    plt.scatter(X[0], X[1], c=colors, alpha=0.7)
    plt.title('Training Data (Red: Class 0, Blue: Class 1)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Decision boundary
    plt.subplot(1, 2, 2)
    
    # Create a mesh
    h = 0.1
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = []
    for i in range(mesh_points.shape[1]):
        point = mesh_points[:, i:i+1]
        pred = nn.predict(point)
        Z.append(pred[0, 0])
    
    Z = np.array(Z).reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    colors = ['red' if y == 0 else 'blue' for y in Y[0]]
    plt.scatter(X[0], X[1], c=colors, alpha=0.7, edgecolors='black')
    plt.title('Decision Boundary Learned by Neural Network')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction')
    
    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main demonstration function showing the significance of backpropagation.
    """
    print("🧠 NEURAL NETWORK BACKPROPAGATION DEMONSTRATION")
    print("📊 This demo shows why backpropagation is crucial for neural network learning")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Demonstrate backpropagation significance
    nn_with_bp, nn_without_bp = demonstrate_backpropagation_significance()
    
    # Visualize learning process
    visualize_learning_process(nn_with_bp)
    
    # Demonstrate on complex problem
    nn_complex = demonstrate_complex_problem()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS ABOUT BACKPROPAGATION")
    print("="*60)
    print("1. WITHOUT backpropagation, neural networks have random weights")
    print("   and cannot learn patterns from data.")
    print("\n2. WITH backpropagation, neural networks can:")
    print("   - Learn complex non-linear patterns (like XOR)")
    print("   - Adjust weights systematically to minimize error")
    print("   - Solve problems that simple linear models cannot")
    print("\n3. The learning process involves:")
    print("   - Forward pass: Computing predictions")
    print("   - Error calculation: Measuring how wrong predictions are")
    print("   - Backward pass: Propagating errors back through layers")
    print("   - Weight updates: Adjusting parameters to reduce error")
    print("\n4. This process repeats until the network learns the pattern!")
    
    print(f"\n✅ Demo completed! Check the generated visualizations:")
    print("📈 backpropagation_analysis.png - Shows learning curves and weight changes")
    print("🎯 decision_boundary.png - Shows how the network separates classes")

if __name__ == "__main__":
    main()
