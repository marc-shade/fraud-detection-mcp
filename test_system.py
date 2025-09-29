#!/usr/bin/env python3
"""
Comprehensive system test for fraud detection MCP
Tests all components and identifies missing pieces
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    results = {}

    modules_to_test = [
        'config',
        'models_validation',
        'training_pipeline',
        'feature_engineering',
        'models.autoencoder',
        'models.gnn_fraud_detector',
        'explainability',
        'security',
        'async_inference',
        'monitoring',
        'benchmarks',
        'server_v2'
    ]

    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            results[module_name] = 'OK'
            print(f'  ✓ {module_name}')
        except ModuleNotFoundError as e:
            results[module_name] = f'MISSING: {e}'
            print(f'  ✗ {module_name} - MISSING')
        except Exception as e:
            results[module_name] = f'ERROR: {e}'
            print(f'  ✗ {module_name} - ERROR: {e}')

    return results

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from config import get_config
        config = get_config()

        assert config.ENVIRONMENT in ['development', 'production']
        assert config.MODEL_DIR.exists()
        assert config.DATA_DIR.exists()

        print(f'  ✓ Configuration loaded successfully')
        print(f'    - Environment: {config.ENVIRONMENT}')
        print(f'    - Model Dir: {config.MODEL_DIR}')
        return True
    except Exception as e:
        print(f'  ✗ Configuration failed: {e}')
        return False

def test_validation():
    """Test input validation"""
    print("\nTesting validation models...")
    try:
        from models_validation import TransactionData, BehavioralData
        from datetime import datetime

        trans = TransactionData(
            transaction_id='test_123',
            user_id='user_456',
            amount=100.50,
            merchant='Test Store',
            location='New York',
            timestamp=datetime.now(),
            payment_method='credit_card'
        )

        print(f'  ✓ Validation models working')
        return True
    except Exception as e:
        print(f'  ✗ Validation failed: {e}')
        return False

def check_file_structure():
    """Check for required files"""
    print("\nChecking file structure...")
    base_dir = Path.cwd()

    required_files = {
        'Core': [
            'config.py',
            'models_validation.py',
            '.env.example',
            'requirements.txt'
        ],
        'ML Components': [
            'training_pipeline.py',
            'feature_engineering.py',
            'models/autoencoder.py',
            'models/gnn_fraud_detector.py'
        ],
        'Advanced Features': [
            'explainability.py',
            'security.py',
            'async_inference.py',
            'monitoring.py'
        ],
        'Server': [
            'server_v2.py',
            'benchmarks.py'
        ]
    }

    missing_files = []

    for category, files in required_files.items():
        print(f'\n  {category}:')
        for file_path in files:
            full_path = base_dir / file_path
            if full_path.exists():
                print(f'    ✓ {file_path}')
            else:
                print(f'    ✗ {file_path} - MISSING')
                missing_files.append(file_path)

    return missing_files

def main():
    print("="*60)
    print("FRAUD DETECTION MCP - SYSTEM TEST")
    print("="*60)

    # Test imports
    import_results = test_imports()

    # Test configuration
    config_ok = test_configuration()

    # Test validation
    validation_ok = test_validation()

    # Check file structure
    missing_files = check_file_structure()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    failed_imports = [k for k, v in import_results.items() if v != 'OK']

    if failed_imports:
        print(f"\n❌ Failed imports ({len(failed_imports)}):")
        for module in failed_imports:
            print(f"  - {module}: {import_results[module]}")

    if missing_files:
        print(f"\n❌ Missing files ({len(missing_files)}):")
        for file in missing_files:
            print(f"  - {file}")

    if not failed_imports and not missing_files and config_ok and validation_ok:
        print("\n✅ ALL TESTS PASSED!")
        print("System is ready for GitHub push.")
        return 0
    else:
        print("\n⚠️ ISSUES FOUND")
        print("Fix the issues above before pushing to GitHub.")
        return 1

if __name__ == "__main__":
    sys.exit(main())