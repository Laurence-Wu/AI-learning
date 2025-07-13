# Reinforcement Learning Paper Reproduction Tutorial

This is a tutorial for reproducing reinforcement learning algorithms from basics to the cutting edge, featuring 8 core exercises that cover the full path from tabular methods to large language model RL.

## Project Structure

```
AI learning/
├── README.md                 # Project introduction
├── requirements.txt          # Dependencies
├── tutorials/                # Tutorial notebooks
│   ├── 01_Q_Learning.ipynb
│   ├── 02_DQN.ipynb
│   ├── 03_Double_DQN.ipynb
│   ├── 04_PPO_Clip.ipynb
│   ├── 05_TD3.ipynb
│   ├── 06_Dreamer_V2.ipynb
│   ├── 07_RLHF.ipynb
│   └── 08_InstructGPT.ipynb
├── solutions/                # Full solutions
│   ├── 01_Q_Learning_Solution.ipynb
│   ├── 02_DQN_Solution.ipynb
│   ├── 03_Double_DQN_Solution.ipynb
│   ├── 04_PPO_Clip_Solution.ipynb
│   ├── 05_TD3_Solution.ipynb
│   ├── 06_Dreamer_V2_Solution.ipynb
│   ├── 07_RLHF_Solution.ipynb
│   └── 08_InstructGPT_Solution.ipynb
└── utils/                    # Utility modules
    ├── environments.py       # Environment wrappers
    ├── networks.py           # Neural network modules
    ├── replay_buffer.py      # Experience replay buffer
    └── visualization.py      # Visualization tools
```

## Core Exercise Overview

1. **Q-learning (1992)** - Tabular RL fundamentals
2. **DQN (2015)** - Deep Q-Network
3. **Double-DQN (2016)** - Reducing overestimation bias
4. **PPO-Clip (2017)** - Proximal Policy Optimization
5. **TD3 (2018)** - Twin Delayed Deep Deterministic Policy Gradient
6. **Dreamer-V2 (2020)** - Model-based RL
7. **RLHF (2017)** - Reinforcement Learning from Human Feedback
8. **InstructGPT (2022)** - Instruction-following LLM fine-tuning

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

## Usage Guide

Each exercise includes two versions:
- **Tutorial version** (`tutorials/`): With theory, exercises, and code skeletons
- **Solution version** (`solutions/`): With full implementation and detailed explanations

Recommended workflow:
1. Read the key sections of the corresponding paper
2. Complete the tutorial version exercises
3. Compare with the solution to verify your understanding
4. Run experiments and analyze results

## Technical Requirements

- Python 3.9+
- PyTorch 2.0+ (Exercises 1-6)
- TensorFlow 2.13+ (Exercises 7-8)
- 8GB+ RAM recommended
- GPU optional but recommended (Exercises 4-8)

## Paper References

For detailed paper lists and citations, please refer to each tutorial notebook.


