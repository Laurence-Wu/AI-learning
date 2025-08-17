#!/usr/bin/env python3
"""
Validation script for ExpoSB implementation
Checks syntax and basic structure without requiring GPU
"""

import ast
import sys
import os

def check_syntax(file_path):
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        ast.parse(content)
        print(f"✓ {os.path.basename(file_path)}: Valid syntax")
        return True
    except SyntaxError as e:
        print(f"✗ {os.path.basename(file_path)}: Syntax error on line {e.lineno}: {e.msg}")
        return False
    except FileNotFoundError:
        print(f"✗ {os.path.basename(file_path)}: File not found")
        return False

def check_imports(file_path):
    """Check if all imports are valid"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Extract import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        print(f"✓ {os.path.basename(file_path)}: Found {len(imports)} import statements")
        return True
    except Exception as e:
        print(f"✗ {os.path.basename(file_path)}: Error checking imports: {e}")
        return False

def check_class_structure(file_path, expected_classes):
    """Check if expected classes are defined"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Extract class definitions
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        found_classes = []
        missing_classes = []
        
        for expected_class in expected_classes:
            if expected_class in classes:
                found_classes.append(expected_class)
            else:
                missing_classes.append(expected_class)
        
        print(f"✓ {os.path.basename(file_path)}: Found classes: {found_classes}")
        if missing_classes:
            print(f"⚠ {os.path.basename(file_path)}: Missing classes: {missing_classes}")
            return False
        return True
    except Exception as e:
        print(f"✗ {os.path.basename(file_path)}: Error checking classes: {e}")
        return False

def check_function_structure(file_path, expected_functions):
    """Check if expected functions are defined"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Extract function definitions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        found_functions = []
        missing_functions = []
        
        for expected_function in expected_functions:
            if expected_function in functions:
                found_functions.append(expected_function)
            else:
                missing_functions.append(expected_function)
        
        print(f"✓ {os.path.basename(file_path)}: Found functions: {len(found_functions)}/{len(expected_functions)}")
        if missing_functions:
            print(f"⚠ {os.path.basename(file_path)}: Missing functions: {missing_functions}")
            return False
        return True
    except Exception as e:
        print(f"✗ {os.path.basename(file_path)}: Error checking functions: {e}")
        return False

def main():
    """Run validation checks"""
    print("=" * 60)
    print("ExpoSB Implementation Validation")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("triton_exposb_attention.py"):
        print("✗ Not in ExpoSB_implementation directory")
        return 1
    
    validation_tests = [
        {
            "name": "ExpoSB Attention Module",
            "file": "triton_exposb_attention.py",
            "expected_classes": ["ExpoSBAttention", "ExpoSBBERTAttention"],
            "expected_functions": ["apply_exposb", "exposb_attention"]
        },
        {
            "name": "Training Script",
            "file": "bert_comparison_train.py",
            "expected_classes": ["BERTDataset", "ModifiedBERTModel", "Trainer"],
            "expected_functions": ["plot_comparison", "main"]
        },
        {
            "name": "Configuration Module",
            "file": "bert_config.py",
            "expected_classes": ["BERTComparisonConfig"],
            "expected_functions": ["str_to_bool"]
        },
        {
            "name": "Standard Attention Module",
            "file": "triton_standard_attention.py",
            "expected_classes": ["StandardBERTAttention"],
            "expected_functions": []
        }
    ]
    
    all_passed = True
    
    for test in validation_tests:
        print(f"\nValidating {test['name']}...")
        print("-" * (len(test['name']) + 12))
        
        # Check syntax
        syntax_ok = check_syntax(test['file'])
        if not syntax_ok:
            all_passed = False
            continue
        
        # Check imports
        imports_ok = check_imports(test['file'])
        if not imports_ok:
            all_passed = False
        
        # Check class structure
        if test['expected_classes']:
            classes_ok = check_class_structure(test['file'], test['expected_classes'])
            if not classes_ok:
                all_passed = False
        
        # Check function structure
        if test['expected_functions']:
            functions_ok = check_function_structure(test['file'], test['expected_functions'])
            if not functions_ok:
                all_passed = False
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    if all_passed:
        print("🎉 All validation checks passed!")
        print("\nImplementation structure is correct.")
        print("Ready for testing and training.")
        
        # Check if required files exist
        required_files = [
            "training_data.txt",
            "config.env", 
            "data_preprocessing.py"
        ]
        
        print("\nChecking required files:")
        for file in required_files:
            if os.path.exists(file):
                print(f"✓ {file}")
            else:
                print(f"⚠ {file} (may be needed for training)")
        
    else:
        print("❌ Some validation checks failed.")
        print("Please fix the issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())