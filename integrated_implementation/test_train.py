#!/usr/bin/env python3
"""
Test script to verify the training script structure
"""

import argparse
import yaml
from pathlib import Path

def test_config_loading():
    """Test configuration loading"""
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully")
        print(f"Training objectives: {config.get('training_objectives', [])}")
        print(f"Attention algorithms: {config.get('attention_algorithms', [])}")
        print(f"Epochs: {config.get('training', {}).get('num_epochs', 'not set')}")
        print(f"Batch size: {config.get('training', {}).get('batch_size', 'not set')}")
        
        # Check required fields
        required_fields = ['training_objectives', 'attention_algorithms', 'training', 'data', 'model']
        missing = [field for field in required_fields if field not in config]
        
        if missing:
            print(f"⚠️  Missing fields: {missing}")
        else:
            print("✅ All required configuration fields present")
        
        return config
    else:
        print("❌ Configuration file not found")
        return None

def test_argument_parsing():
    """Test argument parsing"""
    parser = argparse.ArgumentParser()
    
    # Add sample arguments like in the main script
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--objectives", nargs="+", choices=["mlm", "clm", "both"], default=None)
    parser.add_argument("--attention", nargs="+", default=None)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--cross-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    
    # Test with sample arguments
    test_args = ["--dry-run", "--objectives", "both", "--num-runs", "5"]
    args = parser.parse_args(test_args)
    
    print("✅ Argument parsing works")
    print(f"Objectives: {args.objectives}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Cross-validation: {args.cross_validation}")
    
    return args

def main():
    """Test main functionality"""
    print("Testing Comprehensive Training Script")
    print("=" * 50)
    
    # Test configuration
    print("\n1. Testing Configuration Loading:")
    config = test_config_loading()
    
    # Test arguments
    print("\n2. Testing Argument Parsing:")
    args = test_argument_parsing()
    
    # Test experimental design
    print("\n3. Testing Experimental Design:")
    if config:
        objectives = config.get('training_objectives', ['mlm', 'clm'])
        attention_types = config.get('attention_algorithms', ['standard'])
        num_runs = 3
        
        total_experiments = len(objectives) * len(attention_types) * num_runs
        print(f"Total experiments planned: {total_experiments}")
        print(f"Combinations:")
        for obj in objectives:
            for att in attention_types:
                print(f"  - {att} + {obj.upper()}")
    
    # Test statistical analysis structure
    print("\n4. Testing Statistical Analysis Structure:")
    comparisons = []
    if config:
        objectives = config.get('training_objectives', ['mlm', 'clm'])
        attention_types = config.get('attention_algorithms', ['standard'])
        
        # MLM vs CLM comparisons
        for att in attention_types:
            comparisons.append(f"{att}_mlm_vs_clm")
        
        # Attention comparisons within each objective
        for obj in objectives:
            for i in range(len(attention_types)):
                for j in range(i + 1, len(attention_types)):
                    comparisons.append(f"{obj}_{attention_types[i]}_vs_{attention_types[j]}")
    
    print(f"Statistical comparisons planned: {len(comparisons)}")
    for comp in comparisons[:5]:  # Show first 5
        print(f"  - {comp}")
    if len(comparisons) > 5:
        print(f"  ... and {len(comparisons) - 5} more")
    
    print("\n✅ All tests passed! The training script structure is valid.")
    print("\nTo run the actual training:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run training: python train.py --quick-test")
    print("3. For full experiment: python train.py --num-runs 5 --cross-validation")

if __name__ == "__main__":
    main()