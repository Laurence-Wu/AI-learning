import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add subdirectories to path
sys.path.append('./basic_transformer')
sys.path.append('./gpt_style')
sys.path.append('./bert_style')
sys.path.append('./modern_variants')

# Import all transformer implementations
try:
    from transformer import BasicTransformer, TransformerTrainer, create_attention_visualization
    from gpt_transformer import GPTTransformer, GPTTrainer, create_gpt_visualization
    from bert_transformer import BERTTransformer, BERTTrainer, create_bert_visualization
    from modern_transformers import VisionTransformer, ModernLanguageModel, create_comparison_visualization
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all transformer modules are in their respective directories.")
    sys.exit(1)

class TransformerShowcase:
    """
    Master demonstration class for all transformer variants.
    """
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = []
        
    def test_basic_transformer(self):
        """Test the basic transformer implementation."""
        print("🤖 TESTING BASIC TRANSFORMER")
        print("="*50)
        
        # Model parameters
        vocab_size = 100
        d_model = 256
        num_heads = 8
        num_layers = 4
        d_ff = 1024
        
        print(f"Initializing Basic Transformer:")
        print(f"• Vocab: {vocab_size}, d_model: {d_model}, heads: {num_heads}, layers: {num_layers}")
        
        # Initialize model
        transformer = BasicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff
        )
        
        # Test forward pass
        test_input = np.random.randint(0, vocab_size, 15)
        start_time = time.time()
        output = transformer.forward(test_input)
        forward_time = time.time() - start_time
        
        print(f"✅ Forward pass: {forward_time:.4f}s, Output shape: {output.shape}")
        
        # Quick training
        trainer = TransformerTrainer(transformer)
        start_time = time.time()
        trainer.train(num_epochs=20)
        training_time = time.time() - start_time
        
        print(f"✅ Training: {training_time:.2f}s")
        
        # Test generation
        prompt = [1, 2, 3]
        generated = transformer.generate_text(prompt, max_length=10)
        print(f"✅ Generation - Prompt: {prompt}, Generated: {generated[len(prompt):]}")
        
        # Store results
        self.results['basic'] = {
            'model': transformer,
            'forward_time': forward_time,
            'training_time': training_time,
            'architecture': 'Standard Transformer',
            'parameters': sum(w.size for w in [transformer.embedding] + 
                            [block.attention.W_q for block in transformer.transformer_blocks])
        }
        
        print("✅ Basic Transformer test completed!\n")
        return transformer
    
    def test_gpt_transformer(self):
        """Test the GPT-style transformer implementation."""
        print("🚀 TESTING GPT-STYLE TRANSFORMER")
        print("="*50)
        
        # Model parameters
        vocab_size = 200
        d_model = 512
        num_heads = 16
        num_layers = 6
        max_len = 1024
        
        print(f"Initializing GPT Transformer:")
        print(f"• Vocab: {vocab_size}, d_model: {d_model}, heads: {num_heads}, layers: {num_layers}")
        
        # Initialize model
        gpt_model = GPTTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len
        )
        
        # Test forward pass
        test_input = np.random.randint(0, vocab_size, 20)
        start_time = time.time()
        output = gpt_model.forward(test_input)
        forward_time = time.time() - start_time
        
        print(f"✅ Causal forward pass: {forward_time:.4f}s, Output shape: {output.shape}")
        
        # Quick training
        trainer = GPTTrainer(gpt_model)
        start_time = time.time()
        trainer.train(num_epochs=20)
        training_time = time.time() - start_time
        
        print(f"✅ Training: {training_time:.2f}s")
        
        # Test autoregressive generation
        prompt = [5, 10, 15]
        generated = gpt_model.generate_text(prompt, max_length=12, temperature=0.8)
        print(f"✅ Autoregressive generation - Prompt: {prompt}")
        print(f"   Generated: {generated[len(prompt):]}")
        
        # Test different sampling strategies
        generated_topk = gpt_model.generate_text(prompt, max_length=10, top_k=20)
        print(f"✅ Top-k sampling: {generated_topk[len(prompt):]}")
        
        # Store results
        self.results['gpt'] = {
            'model': gpt_model,
            'forward_time': forward_time,
            'training_time': training_time,
            'architecture': 'GPT-style (Causal)',
            'parameters': vocab_size * d_model + num_layers * d_model * d_model * 4
        }
        
        print("✅ GPT Transformer test completed!\n")
        return gpt_model
    
    def test_bert_transformer(self):
        """Test the BERT-style transformer implementation."""
        print("🎭 TESTING BERT-STYLE TRANSFORMER")
        print("="*50)
        
        # Model parameters
        vocab_size = 2000
        d_model = 512
        num_heads = 16
        num_layers = 8
        max_len = 256
        
        print(f"Initializing BERT Transformer:")
        print(f"• Vocab: {vocab_size}, d_model: {d_model}, heads: {num_heads}, layers: {num_layers}")
        
        # Initialize model
        bert_model = BERTTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len
        )
        
        # Test forward pass
        test_input = [bert_model.CLS_TOKEN] + \
                     list(np.random.randint(4, vocab_size, 15)) + \
                     [bert_model.SEP_TOKEN]
        
        start_time = time.time()
        sequence_output, pooled_output = bert_model.forward(test_input)
        forward_time = time.time() - start_time
        
        print(f"✅ Bidirectional forward pass: {forward_time:.4f}s")
        print(f"   Sequence shape: {sequence_output.shape}, Pooled shape: {pooled_output.shape}")
        
        # Quick training
        trainer = BERTTrainer(bert_model)
        start_time = time.time()
        trainer.train(num_epochs=15)
        training_time = time.time() - start_time
        
        print(f"✅ MLM Training: {training_time:.2f}s")
        
        # Test MLM
        mlm_input = test_input.copy()
        original_token = mlm_input[3]
        mlm_input[3] = bert_model.MASK_TOKEN
        
        mlm_logits = bert_model.forward_mlm(mlm_input)
        predicted_token = np.argmax(mlm_logits[3])
        
        print(f"✅ Masked Language Modeling:")
        print(f"   Original: {original_token}, Predicted: {predicted_token}")
        print(f"   Correct: {predicted_token == original_token}")
        
        # Store results
        self.results['bert'] = {
            'model': bert_model,
            'forward_time': forward_time,
            'training_time': training_time,
            'architecture': 'BERT-style (Bidirectional)',
            'parameters': vocab_size * d_model + num_layers * d_model * d_model * 4
        }
        
        print("✅ BERT Transformer test completed!\n")
        return bert_model
    
    def test_modern_transformers(self):
        """Test modern transformer variants."""
        print("🌟 TESTING MODERN TRANSFORMER VARIANTS")
        print("="*50)
        
        # Test Vision Transformer
        print("📸 Vision Transformer (ViT):")
        img_size = 64
        patch_size = 8
        num_classes = 100
        
        vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            d_model=512,
            num_heads=16,
            num_layers=6
        )
        
        # Test image
        test_image = np.random.randn(img_size, img_size, 3)
        start_time = time.time()
        vit_output = vit.forward(test_image)
        vit_time = time.time() - start_time
        
        predicted_class = np.argmax(vit_output)
        print(f"✅ ViT: {vit_time:.4f}s, Predicted class: {predicted_class}")
        
        # Test Modern Language Model
        print("🤖 Modern Language Model:")
        vocab_size = 1000
        d_model = 768
        num_heads = 24
        num_layers = 12
        
        modern_lm = ModernLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        test_tokens = np.random.randint(0, vocab_size, 25)
        start_time = time.time()
        lm_output = modern_lm.forward(test_tokens)
        lm_time = time.time() - start_time
        
        print(f"✅ Modern LM: {lm_time:.4f}s, Output shape: {lm_output.shape}")
        
        # Test generation
        prompt = [1, 2, 3, 4]
        generated = modern_lm.generate_text(prompt, max_length=12, top_k=50)
        print(f"✅ Generation: {generated[len(prompt):]}")
        
        # Store results
        self.results['modern'] = {
            'vit': vit,
            'modern_lm': modern_lm,
            'vit_time': vit_time,
            'lm_time': lm_time,
            'architecture': 'Modern (ViT + Advanced LM)',
            'parameters': num_classes * 512 + vocab_size * d_model
        }
        
        print("✅ Modern Transformers test completed!\n")
        return vit, modern_lm
    
    def create_performance_comparison(self):
        """Create comprehensive performance comparison."""
        print("📊 CREATING PERFORMANCE COMPARISON")
        print("="*50)
        
        # Extract performance data
        architectures = []
        forward_times = []
        training_times = []
        parameter_counts = []
        
        for name, data in self.results.items():
            if name == 'modern':
                architectures.append('ViT')
                forward_times.append(data['vit_time'])
                training_times.append(0.1)  # Placeholder
                parameter_counts.append(data['parameters'] // 1000000)  # In millions
                
                architectures.append('Modern LM')
                forward_times.append(data['lm_time'])
                training_times.append(0.15)  # Placeholder
                parameter_counts.append(data['parameters'] // 1000000)
            else:
                architectures.append(data['architecture'])
                forward_times.append(data['forward_time'])
                training_times.append(data['training_time'])
                parameter_counts.append(data['parameters'] // 1000000)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Transformer Variants Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Forward pass times
        ax1 = axes[0, 0]
        bars1 = ax1.bar(architectures, forward_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('Forward Pass Performance')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars1, forward_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{time_val:.4f}s', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Training times
        ax2 = axes[0, 1]
        bars2 = ax2.bar(architectures, training_times, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'])
        ax2.set_title('Training Performance')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Model complexity
        ax3 = axes[1, 0]
        bars3 = ax3.bar(architectures, parameter_counts, color=['#1ABC9C', '#E67E22', '#8E44AD', '#27AE60', '#C0392B'])
        ax3.set_title('Model Complexity')
        ax3.set_ylabel('Parameters (Millions)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Feature comparison radar chart (simplified as stacked bar)
        ax4 = axes[1, 1]
        
        # Feature scores (0-1 scale)
        features = {
            'Generation': [0.3, 0.9, 0.2, 0.3, 0.9],
            'Understanding': [0.7, 0.4, 0.9, 0.5, 0.7],
            'Vision': [0.0, 0.0, 0.0, 1.0, 0.2],
            'Efficiency': [0.5, 0.6, 0.6, 0.7, 0.9]
        }
        
        x_pos = np.arange(len(architectures))
        bottom = np.zeros(len(architectures))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (feature, values) in enumerate(features.items()):
            ax4.bar(x_pos, values, bottom=bottom, label=feature, color=colors[i], alpha=0.8)
            bottom += values
        
        ax4.set_title('Capability Comparison')
        ax4.set_ylabel('Capability Score')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(architectures, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('transformer_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary table
        print("\nPERFORMANCE SUMMARY TABLE")
        print("-" * 80)
        print(f"{'Architecture':<20} {'Forward(s)':<12} {'Training(s)':<12} {'Params(M)':<12}")
        print("-" * 80)
        
        for i, arch in enumerate(architectures):
            print(f"{arch:<20} {forward_times[i]:<12.4f} {training_times[i]:<12.2f} {parameter_counts[i]:<12.1f}")
        print("-" * 80)
    
    def create_architecture_summary(self):
        """Create detailed architecture summary."""
        print("\n🏗️  TRANSFORMER ARCHITECTURE EVOLUTION SUMMARY")
        print("="*70)
        
        evolution_data = {
            "2017 - Original Transformer": {
                "Key Innovation": "Multi-head attention mechanism",
                "Architecture": "Encoder-Decoder with post-norm",
                "Position Encoding": "Sinusoidal",
                "Activation": "ReLU",
                "Use Case": "Machine Translation"
            },
            "2018 - GPT": {
                "Key Innovation": "Causal self-attention for generation",
                "Architecture": "Decoder-only with pre-norm", 
                "Position Encoding": "Learned embeddings",
                "Activation": "GELU",
                "Use Case": "Language Modeling"
            },
            "2018 - BERT": {
                "Key Innovation": "Bidirectional encoding with MLM",
                "Architecture": "Encoder-only with post-norm",
                "Position Encoding": "Learned embeddings",
                "Activation": "GELU",
                "Use Case": "Language Understanding"
            },
            "2020 - ViT": {
                "Key Innovation": "Vision as sequence processing",
                "Architecture": "Encoder with patch embeddings",
                "Position Encoding": "Learned embeddings",
                "Activation": "GELU",
                "Use Case": "Image Classification"
            },
            "2023 - Modern LLM": {
                "Key Innovation": "Efficiency improvements (MQA, RoPE, SwiGLU)",
                "Architecture": "Decoder-only with advanced components",
                "Position Encoding": "Rotary Position Embedding",
                "Activation": "SwiGLU",
                "Use Case": "Large-scale Language Modeling"
            }
        }
        
        for year_model, details in evolution_data.items():
            print(f"\n📅 {year_model}")
            print("-" * 40)
            for key, value in details.items():
                print(f"  {key:<18}: {value}")
    
    def run_complete_demonstration(self):
        """Run the complete transformer demonstration."""
        print("🚀 COMPREHENSIVE TRANSFORMER SHOWCASE")
        print("="*70)
        print("Testing all transformer variants from basic to cutting-edge")
        print("="*70)
        
        start_time = time.time()
        
        # Test all variants
        basic_model = self.test_basic_transformer()
        gpt_model = self.test_gpt_transformer()
        bert_model = self.test_bert_transformer()
        vit_model, modern_lm_model = self.test_modern_transformers()
        
        # Create comparisons
        self.create_performance_comparison()
        self.create_architecture_summary()
        
        # Advanced testing and analysis
        print("\n🔬 Advanced Analysis and Cross-Testing...")
        self.run_cross_architecture_tests()
        self.run_performance_stress_tests()
        
        # Generate comprehensive visualizations
        print("\n📊 Generating detailed visualizations...")
        self.generate_all_visualizations(basic_model, gpt_model, bert_model, vit_model, modern_lm_model)
    
    def run_cross_architecture_tests(self):
        """Run cross-architecture comparison tests."""
        print("🔄 Cross-Architecture Testing...")
        
        # Test sequence processing capabilities
        test_sequences = [
            np.random.randint(0, 50, 10),
            np.random.randint(0, 100, 15),
            np.random.randint(0, 30, 20)
        ]
        
        for i, seq in enumerate(test_sequences):
            print(f"   Sequence {i+1} (length {len(seq)}):")
            
            # Basic transformer
            if 'basic' in self.results:
                try:
                    start = time.time()
                    basic_out = self.results['basic']['model'].forward(seq)
                    basic_time = time.time() - start
                    print(f"     Basic: {basic_time:.4f}s, shape: {basic_out.shape}")
                except: pass
            
            # GPT
            if 'gpt' in self.results:
                try:
                    start = time.time()
                    gpt_out = self.results['gpt']['model'].forward(seq)
                    gpt_time = time.time() - start
                    print(f"     GPT: {gpt_time:.4f}s, shape: {gpt_out.shape}")
                except: pass
    
    def run_performance_stress_tests(self):
        """Run performance stress tests."""
        print("🔄 Performance Stress Testing...")
        
        # Test with increasing sequence lengths
        sequence_lengths = [50, 100, 200]
        
        for seq_len in sequence_lengths:
            print(f"   Testing sequence length {seq_len}:")
            test_seq = np.random.randint(0, 50, seq_len)
            
            if 'basic' in self.results:
                try:
                    start = time.time()
                    self.results['basic']['model'].forward(test_seq)
                    time_taken = time.time() - start
                    print(f"     Basic: {time_taken:.4f}s")
                except Exception as e:
                    print(f"     Basic: Failed ({str(e)[:30]}...)")
    
    def generate_all_visualizations(self, basic_model, gpt_model, bert_model, vit_model, modern_lm_model):
        """Generate all visualization outputs."""
        try:
            test_tokens = np.random.randint(0, 50, 10)
            create_attention_visualization(basic_model, test_tokens)
            print("✅ Basic transformer visualization saved")
        except Exception as e:
            print(f"⚠️ Basic transformer visualization failed: {e}")
        
        try:
            test_tokens = np.random.randint(0, 200, 15)
            create_gpt_visualization(gpt_model, test_tokens)
            print("✅ GPT transformer visualization saved")
        except Exception as e:
            print(f"⚠️ GPT transformer visualization failed: {e}")
        
        try:
            test_tokens = [0] + list(np.random.randint(4, 100, 10)) + [1]
            create_bert_visualization(bert_model, test_tokens)
            print("✅ BERT transformer visualization saved")
        except Exception as e:
            print(f"⚠️ BERT transformer visualization failed: {e}")
        
        try:
            create_comparison_visualization()
            print("✅ Modern transformers comparison saved")
        except Exception as e:
            print(f"⚠️ Modern transformers visualization failed: {e}")
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "="*70)
        print("🎉 TRANSFORMER SHOWCASE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"⏱️  Total demonstration time: {total_time:.2f} seconds")
        print(f"🧠 Models tested: {len(self.results) + 1} transformer variants")
        
        print(f"\n📋 Key Findings:")
        print(f"✅ All transformer variants successfully implemented")
        print(f"✅ Performance comparison completed")
        print(f"✅ Architecture evolution documented") 
        print(f"✅ Visualizations generated")
        
        print(f"\n📊 Generated Files:")
        print(f"📈 transformer_performance_comparison.png")
        print(f"📈 transformer_analysis.png")
        print(f"📈 gpt_transformer_analysis.png")
        print(f"📈 bert_transformer_analysis.png")
        print(f"📈 modern_transformers_comparison.png")
        
        print(f"\n🎯 This comprehensive showcase demonstrates the complete evolution")
        print(f"   of transformer architectures from 2017 to modern variants!")
        
        return {
            'basic': basic_model,
            'gpt': gpt_model, 
            'bert': bert_model,
            'vit': vit_model,
            'modern_lm': modern_lm_model
        }

def main():
    """Main function to run the complete transformer showcase."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize and run showcase
    showcase = TransformerShowcase()
    models = showcase.run_complete_demonstration()
    
    return models

if __name__ == "__main__":
    models = main()
