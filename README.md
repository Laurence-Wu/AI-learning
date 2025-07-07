# AI Learning Repository

A comprehensive collection of AI/ML implementations, research papers, and learning materials organized for educational purposes.

## 📁 Repository Structure

### 01_Fundamentals/
Core machine learning concepts and implementations
- **Neural_Networks/**: Basic neural network implementation with backpropagation
  - `neural_network.py`: Core neural network class
  - `backpropagation_demo.py`: Demonstration of backpropagation algorithm
  - `large_dataset_test.py`: Testing on larger datasets

### 02_Tokenization/
Text processing and tokenization implementations
- **BPE_Implementation/**: Byte Pair Encoding tokenization
  - `Byte_Pair_Encoding.py`: Main BPE implementation
  - `bpe_demo.py`: Usage examples and demonstrations
  - `learn_bpe.py`: BPE learning algorithm
  - `context.py`: Context handling utilities

### 03_Transformers/
Transformer architecture implementations
- **Basic_Implementation/**: Fundamental transformer implementation
- **BERT_Style/**: BERT-style bidirectional transformer
- **GPT_Style/**: GPT-style autoregressive transformer  
- **Modern_Variants/**: Contemporary transformer architectures

### 04_Research_Papers/
Curated collection of important research papers
- **Attention_Mechanisms/**: Papers on attention and related mechanisms
- **Fine_Tuning/**: Papers on model fine-tuning techniques

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see individual requirements.txt files)

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd AI-learning
   ```

2. Install dependencies for each module:
   ```bash
   # For neural networks
   cd 01_Fundamentals/Neural_Networks
   pip install -r requirements.txt
   
   # For transformers (if requirements exist)
   cd ../../03_Transformers
   pip install torch transformers numpy
   ```

### Usage Examples

#### Neural Networks
```bash
cd 01_Fundamentals/Neural_Networks
python backpropagation_demo.py
```

#### BPE Tokenization
```bash
cd 02_Tokenization/BPE_Implementation
python bpe_demo.py
```

#### Transformers
```bash
cd 03_Transformers
python transformer_demo.py
```

## 📚 Learning Path

1. **Start with Fundamentals**: Begin with neural networks to understand backpropagation
2. **Text Processing**: Learn tokenization with BPE implementation
3. **Advanced Architectures**: Explore transformer implementations
4. **Research**: Read papers to understand cutting-edge techniques

## 🔬 Research Papers Included

### Attention Mechanisms
- Contextual Position Encoding: Learning to Count What's Important
- Forgetting Transformer: Softmax Attention with a Forget Gate
- Scaling Stick-Breaking Attention: An Efficient Implementation and In-Depth Study

### Fine-Tuning
- The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs

## 🤝 Contributing

Feel free to contribute by:
- Adding new implementations
- Improving existing code
- Adding more research papers
- Enhancing documentation

## 📝 License

This repository is for educational purposes. Please respect the licenses of individual research papers and implementations.

## 🏷️ Tags
`machine-learning` `deep-learning` `transformers` `neural-networks` `tokenization` `bert` `gpt` `attention-mechanism` `ai` `python`
