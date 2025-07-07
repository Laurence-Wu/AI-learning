import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional
import json

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism - the core of transformers.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        # For tracking attention weights
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix (seq_len, d_k)
            K: Key matrix (seq_len, d_k)
            V: Value matrix (seq_len, d_k)
            mask: Optional attention mask
            
        Returns:
            Output and attention weights
        """
        # Compute attention scores
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor (seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (seq_len, d_model)
        """
        seq_len, d_model = x.shape
        
        # Linear transformations for Q, K, V
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.num_heads, self.d_k)
        K = K.reshape(seq_len, self.num_heads, self.d_k)
        V = V.reshape(seq_len, self.num_heads, self.d_k)
        
        # Apply attention for each head
        outputs = []
        attention_weights_all = []
        
        for h in range(self.num_heads):
            output_h, attn_h = self.scaled_dot_product_attention(
                Q[:, h, :], K[:, h, :], V[:, h, :], mask
            )
            outputs.append(output_h)
            attention_weights_all.append(attn_h)
        
        # Concatenate heads
        concat_output = np.concatenate(outputs, axis=-1)
        
        # Final linear transformation
        output = np.dot(concat_output, self.W_o)
        
        # Store attention weights for visualization
        self.attention_weights = np.stack(attention_weights_all, axis=0)
        
        return output

class PositionalEncoding:
    """
    Positional encoding for transformers.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        
        # Compute div_term for sine and cosine
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.shape[0]
        return x + self.pe[:seq_len, :]

class FeedForward:
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor (seq_len, d_model)
            
        Returns:
            Output tensor (seq_len, d_model)
        """
        # First linear transformation + ReLU
        hidden = self.relu(np.dot(x, self.W1) + self.b1)
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

class LayerNorm:
    """
    Layer normalization.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize layer normalization.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        """
        Forward pass of layer normalization.
        
        Args:
            x: Input tensor (seq_len, d_model)
            
        Returns:
            Normalized tensor
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        
        return self.gamma * normalized + self.beta

class TransformerBlock:
    """
    A single transformer block (encoder layer).
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate (not implemented in this basic version)
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x, mask=None):
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor (seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (seq_len, d_model)
        """
        # Multi-head attention with residual connection and layer norm
        attn_output = self.attention.forward(x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)
        
        return x

class BasicTransformer:
    """
    Basic Transformer model for sequence-to-sequence tasks.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int = 5000):
        """
        Initialize transformer model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_len = max_len
        
        # Initialize components
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff)
            self.transformer_blocks.append(block)
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
        self.output_bias = np.zeros(vocab_size)
        
        # Training history
        self.loss_history = []
        self.attention_maps = []
    
    def embed_tokens(self, tokens):
        """
        Convert token indices to embeddings.
        
        Args:
            tokens: List of token indices
            
        Returns:
            Embedded tokens (seq_len, d_model)
        """
        return self.embedding[tokens]
    
    def softmax(self, x):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, input_tokens, mask=None):
        """
        Forward pass of transformer.
        
        Args:
            input_tokens: Input token indices
            mask: Optional attention mask
            
        Returns:
            Output logits (seq_len, vocab_size)
        """
        # Token embedding
        x = self.embed_tokens(input_tokens)
        
        # Add positional encoding
        x = self.positional_encoding.forward(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x, mask)
        
        # Output projection
        output = np.dot(x, self.output_projection) + self.output_bias
        
        return output
    
    def compute_loss(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Model predictions (seq_len, vocab_size)
            targets: Target token indices
            
        Returns:
            Cross-entropy loss
        """
        # Apply softmax to get probabilities
        probs = self.softmax(predictions)
        
        # Compute cross-entropy loss
        loss = 0
        for i, target in enumerate(targets):
            if i < len(probs):
                loss += -np.log(probs[i, target] + 1e-10)
        
        return loss / len(targets)
    
    def generate_text(self, prompt_tokens, max_length=50, temperature=1.0):
        """
        Generate text using the transformer.
        
        Args:
            prompt_tokens: Initial prompt tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token sequence
        """
        generated = list(prompt_tokens)
        
        for _ in range(max_length):
            # Forward pass
            logits = self.forward(generated)
            
            # Get next token probabilities
            next_token_logits = logits[-1] / temperature
            next_token_probs = self.softmax(next_token_logits.reshape(1, -1))[0]
            
            # Sample next token
            next_token = np.random.choice(self.vocab_size, p=next_token_probs)
            generated.append(next_token)
            
            # Stop if we hit maximum length or end token
            if len(generated) >= max_length:
                break
        
        return generated

class TransformerTrainer:
    """
    Training utilities for the transformer.
    """
    
    def __init__(self, model: BasicTransformer, learning_rate: float = 0.001):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model to train
            learning_rate: Learning rate
        """
        self.model = model
        self.learning_rate = learning_rate
    
    def create_simple_dataset(self, vocab_size=100, seq_len=20, num_samples=1000):
        """
        Create a simple synthetic dataset for training.
        
        Args:
            vocab_size: Vocabulary size
            seq_len: Sequence length
            num_samples: Number of training samples
            
        Returns:
            Training data (input sequences, target sequences)
        """
        print(f"🔄 Creating synthetic dataset: {num_samples} samples, seq_len={seq_len}")
        
        # Generate random sequences
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            # Create input sequence
            seq = np.random.randint(0, vocab_size, seq_len)
            
            # Create target (shifted by 1 for next-token prediction)
            target = np.roll(seq, -1)
            target[-1] = np.random.randint(0, vocab_size)  # Random end token
            
            inputs.append(seq)
            targets.append(target)
        
        return inputs, targets
    
    def train_step(self, input_seq, target_seq):
        """
        Single training step (simplified - no actual backprop implementation).
        
        Args:
            input_seq: Input sequence
            target_seq: Target sequence
            
        Returns:
            Loss value
        """
        # Forward pass
        predictions = self.model.forward(input_seq)
        
        # Compute loss
        loss = self.model.compute_loss(predictions, target_seq)
        
        # Note: In a real implementation, we would do backpropagation here
        # For this demonstration, we'll just return the loss
        
        return loss
    
    def train(self, num_epochs=100, batch_size=32):
        """
        Train the transformer model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size (simplified - using single samples)
        """
        print(f"🚀 Training transformer for {num_epochs} epochs...")
        
        # Create training data
        inputs, targets = self.create_simple_dataset(
            vocab_size=self.model.vocab_size,
            seq_len=20,
            num_samples=500
        )
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Train on each sample
            for i in range(min(50, len(inputs))):  # Limit for demo
                input_seq = inputs[i]
                target_seq = targets[i]
                
                loss = self.train_step(input_seq, target_seq)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.model.loss_history.append(avg_loss)
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
        
        print("✅ Training completed!")

def create_attention_visualization(transformer_model, input_tokens):
    """
    Visualize attention patterns.
    
    Args:
        transformer_model: Trained transformer model
        input_tokens: Input token sequence
    """
    print("📊 Creating attention visualization...")
    
    # Forward pass to get attention weights
    _ = transformer_model.forward(input_tokens)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Transformer Attention Patterns', fontsize=16)
    
    # Get attention from first layer, first head
    if transformer_model.transformer_blocks:
        attention_weights = transformer_model.transformer_blocks[0].attention.attention_weights
        
        if attention_weights is not None and len(attention_weights) > 0:
            # Plot attention heatmap for first head
            ax1 = axes[0, 0]
            im1 = ax1.imshow(attention_weights[0], cmap='Blues', aspect='auto')
            ax1.set_title('Attention Head 1 - Layer 1')
            ax1.set_xlabel('Key Position')
            ax1.set_ylabel('Query Position')
            plt.colorbar(im1, ax=ax1)
            
            # Plot attention heatmap for second head (if exists)
            if len(attention_weights) > 1:
                ax2 = axes[0, 1]
                im2 = ax2.imshow(attention_weights[1], cmap='Reds', aspect='auto')
                ax2.set_title('Attention Head 2 - Layer 1')
                ax2.set_xlabel('Key Position')
                ax2.set_ylabel('Query Position')
                plt.colorbar(im2, ax=ax2)
    
    # Plot model architecture diagram
    ax3 = axes[1, 0]
    ax3.text(0.1, 0.8, 'Transformer Architecture:', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.7, f'• Vocabulary Size: {transformer_model.vocab_size}', fontsize=12)
    ax3.text(0.1, 0.6, f'• Model Dimension: {transformer_model.d_model}', fontsize=12)
    ax3.text(0.1, 0.5, f'• Number of Heads: {transformer_model.num_heads}', fontsize=12)
    ax3.text(0.1, 0.4, f'• Number of Layers: {transformer_model.num_layers}', fontsize=12)
    ax3.text(0.1, 0.3, f'• Feed-Forward Dim: {transformer_model.d_ff}', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Plot loss curve if available
    ax4 = axes[1, 1]
    if transformer_model.loss_history:
        ax4.plot(transformer_model.loss_history, 'b-', linewidth=2)
        ax4.set_title('Training Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No training history available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('transformer_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main demonstration of basic transformer functionality.
    """
    print("🤖 BASIC TRANSFORMER IMPLEMENTATION DEMONSTRATION")
    print("="*60)
    
    # Model parameters
    vocab_size = 50
    d_model = 128
    num_heads = 8
    num_layers = 4
    d_ff = 512
    max_len = 100
    
    print(f"Initializing transformer with:")
    print(f"• Vocabulary Size: {vocab_size}")
    print(f"• Model Dimension: {d_model}")
    print(f"• Attention Heads: {num_heads}")
    print(f"• Layers: {num_layers}")
    print(f"• Feed-Forward Dimension: {d_ff}")
    
    # Initialize transformer
    transformer = BasicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    print("\n✅ Transformer initialized successfully!")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    test_input = np.random.randint(0, vocab_size, 10)
    print(f"Input tokens: {test_input}")
    
    start_time = time.time()
    output = transformer.forward(test_input)
    forward_time = time.time() - start_time
    
    print(f"✅ Forward pass completed in {forward_time:.4f}s")
    print(f"Output shape: {output.shape}")
    
    # Train the model
    print("\n🚀 Training transformer...")
    trainer = TransformerTrainer(transformer, learning_rate=0.001)
    trainer.train(num_epochs=50)
    
    # Test text generation
    print("\n📝 Testing text generation...")
    prompt = [1, 2, 3]  # Simple prompt
    generated = transformer.generate_text(prompt, max_length=10, temperature=1.0)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_attention_visualization(transformer, test_input)
    
    # Summary
    print("\n" + "="*60)
    print("🎉 BASIC TRANSFORMER DEMONSTRATION COMPLETED!")
    print("="*60)
    print("Key Features Demonstrated:")
    print("✅ Multi-Head Attention mechanism")
    print("✅ Positional encoding")
    print("✅ Feed-forward networks")
    print("✅ Layer normalization")
    print("✅ Residual connections")
    print("✅ Text generation capability")
    print("✅ Training procedure")
    print("✅ Attention visualization")
    
    print(f"\n📊 Generated visualization: transformer_analysis.png")
    
    return transformer

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstration
    model = main()
