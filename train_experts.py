#!/usr/bin/env python3
"""
Train Expert Models for FoNE Multi-Operation System

Usage:
    # Train all experts sequentially
    python train_experts.py --train_all
    
    # Train specific expert
    python train_experts.py --operation addition
    
    # Test the router with trained experts
    python train_experts.py --test_router
"""

import argparse
import os
import sys
import logging
import torch
import wandb

# Add parent directory to path to import from main repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model_utils import load_model_and_tokenizer
from utils.logger_utils import setup_logger, get_embedding_dim
from number_encoders.FNE import FNE
from train.train_pipeline import create_dataloader_and_train

# Import our new expert system (save this as expert_system.py)
from expert_system import ExpertModelManager, ExpertRouter, ExpressionParser

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERT_CONFIGS = {
    'addition': {
        'dataset': 'Onlydrinkwater/1000addition',
        'description': 'Single-digit to 3-digit addition'
    },
    'subtraction': {
        'dataset': 'Onlydrinkwater/1000addition',  # TODO: Change when subtraction dataset available
        'description': 'Subtraction operations (using addition dataset for now)'
    },
    'multiplication': {
        'dataset': 'Onlydrinkwater/1000addition',  # TODO: Change when multiplication dataset available
        'description': 'Multiplication operations (using addition dataset for now)'
    },
    'division': {
        'dataset': 'Onlydrinkwater/1000addition',  # TODO: Change when division dataset available
        'description': 'Division operations (using addition dataset for now)'
    }
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_base_args():
    """Get base training arguments."""
    base_args = argparse.Namespace(
        batch_size=32,
        epochs=5,
        int_digit_len=10,
        frac_digit_len=0,
        len_gen_size=0,
        lr=5e-4,
        model='meta-llama/Llama-3.2-1B-Instruct',
        train_from_scratch=True,
        model_size_level=4,
        num_train_samples=10000,
        num_test_samples=2000,
        seed=42,
        method='fne',
        period_base_list=[10.0],
        clip=True,
        add_linear=True,
        scheduler_name='cosine',
        use_digit_wise_tokenizer=False,
    )
    return base_args


def train_expert(operation, args_override=None):
    """Train a single expert model."""
    
    # Get base args and override if needed
    args = get_base_args()
    if args_override:
        for key, value in args_override.items():
            setattr(args, key, value)
    
    # Set operation-specific config
    config = EXPERT_CONFIGS[operation]
    args.dataset = config['dataset']
    args.name = f'expert_{operation}'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup logging
    output_folder = f"result/experts/{operation}"
    os.makedirs(output_folder, exist_ok=True)
    setup_logger(output_folder)
    
    logging.info("="*80)
    logging.info(f"Training {operation.upper()} Expert")
    logging.info(f"Description: {config['description']}")
    logging.info(f"Dataset: {config['dataset']}")
    logging.info("="*80)
    logging.info(f"Arguments: {args}")
    
    # Initialize WandB
    wandb.init(
        project="FoNE_Experts",
        name=f"{operation}_expert",
        config=vars(args)
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        cache_dir="./hg_cache",
        device=device,
        train_from_scratch=args.train_from_scratch,
        size_level=args.model_size_level,
        use_digit_wise_tokenizer=args.use_digit_wise_tokenizer
    )
    
    # Initialize FNE
    embedding_dim = get_embedding_dim(model)
    fne = FNE(
        embedding_dim,
        int_digit_len=args.int_digit_len,
        frac_digit_len=args.frac_digit_len,
        period_base_list=args.period_base_list,
        add_linear=args.add_linear,
        device=device
    ).to(device)
    
    # Train the model
    create_dataloader_and_train(args, model, tokenizer, device)
    
    # Save the expert
    manager = ExpertModelManager()
    
    # Extract final accuracy from logs (you might want to return this from training)
    # For now, we'll just save with placeholder
    manager.save_expert(model, fne, operation, metadata={
        'dataset': config['dataset'],
        'description': config['description'],
        'epochs': args.epochs,
        'model_size_level': args.model_size_level,
        'num_train_samples': args.num_train_samples
    })
    
    wandb.finish()
    logging.info(f"✓ {operation} expert trained and saved!")


def train_all_experts(operations=None):
    """Train all expert models sequentially."""
    
    if operations is None:
        operations = list(EXPERT_CONFIGS.keys())
    
    print("\n" + "="*80)
    print("TRAINING ALL EXPERT MODELS")
    print("="*80)
    print(f"Operations to train: {', '.join(operations)}")
    print(f"Total experts: {len(operations)}")
    print("="*80 + "\n")
    
    for i, operation in enumerate(operations, 1):
        print(f"\n[{i}/{len(operations)}] Starting {operation} expert training...")
        try:
            train_expert(operation)
            print(f"✓ {operation} expert completed successfully!")
        except Exception as e:
            print(f"✗ Error training {operation} expert: {e}")
            logging.error(f"Failed to train {operation}: {e}", exc_info=True)
            continue
    
    print("\n" + "="*80)
    print("ALL EXPERTS TRAINED!")
    print("="*80)


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_single_expert(operation, test_expressions):
    """Test a single expert on simple expressions."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load base model architecture
    args = get_base_args()
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        cache_dir="./hg_cache",
        device=device,
        train_from_scratch=True,
        size_level=args.model_size_level
    )
    
    # Load FNE
    embedding_dim = get_embedding_dim(model)
    fne = FNE(
        embedding_dim,
        int_digit_len=args.int_digit_len,
        frac_digit_len=args.frac_digit_len,
        period_base_list=args.period_base_list,
        add_linear=args.add_linear,
        device=device
    ).to(device)
    
    # Load the trained expert
    manager = ExpertModelManager()
    model, fne, metadata = manager.load_expert(model, fne, operation, device)
    
    print(f"\n{'='*60}")
    print(f"Testing {operation.upper()} Expert")
    print(f"Metadata: {metadata}")
    print(f"{'='*60}\n")
    
    # Initialize router
    router = ExpertRouter(manager, model, fne, device)
    router.experts[operation] = {'model': model, 'fne': fne, 'metadata': metadata}
    
    # Test each expression
    results = []
    for expr in test_expressions:
        try:
            result = router.solve_expression(expr, tokenizer, use_pemdas=False,
                                            int_digit_len=args.int_digit_len,
                                            frac_digit_len=args.frac_digit_len)
            
            # Calculate expected result for simple expressions
            expected = eval(expr.replace('=', ''))
            correct = abs(result - expected) < 0.5
            
            results.append({
                'expression': expr,
                'predicted': result,
                'expected': expected,
                'correct': correct
            })
            
            status = "✓" if correct else "✗"
            print(f"{status} {expr} = {result:.2f} (expected: {expected})")
            
        except Exception as e:
            print(f"✗ {expr} - Error: {e}")
            results.append({
                'expression': expr,
                'predicted': None,
                'expected': eval(expr.replace('=', '')),
                'correct': False
            })
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) * 100
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    print(f"{'='*60}\n")
    
    return results


def test_router_multi_operation():
    """Test the router on multi-operation expressions."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load base architectures
    args = get_base_args()
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        cache_dir="./hg_cache",
        device=device,
        train_from_scratch=True,
        size_level=args.model_size_level
    )
    
    embedding_dim = get_embedding_dim(model)
    fne = FNE(
        embedding_dim,
        int_digit_len=args.int_digit_len,
        frac_digit_len=args.frac_digit_len,
        period_base_list=args.period_base_list,
        add_linear=args.add_linear,
        device=device
    ).to(device)
    
    # Initialize router
    manager = ExpertModelManager()
    router = ExpertRouter(manager, model, fne, device)
    
    # Test cases (left-to-right evaluation, no PEMDAS)
    test_cases = [
        ("100+50-25", 125),      # (100+50)-25 = 150-25 = 125
        ("50-20+10", 40),        # (50-20)+10 = 30+10 = 40
        ("10+20+30", 60),        # (10+20)+30 = 30+30 = 60
        ("100-10-20-30", 40),    # ((100-10)-20)-30 = 40
    ]
    
    print("\n" + "="*80)
    print("TESTING ROUTER WITH MULTI-OPERATION EXPRESSIONS")
    print("Mode: Left-to-right evaluation (no PEMDAS)")
    print("="*80 + "\n")
    
    results = []
    for expr, expected in test_cases:
        try:
            result = router.solve_expression(expr, tokenizer, use_pemdas=False,
                                            int_digit_len=args.int_digit_len,
                                            frac_digit_len=args.frac_digit_len)
            
            correct = abs(result - expected) < 0.5
            status = "✓" if correct else "✗"
            
            print(f"{status} {expr} = {result:.2f} (expected: {expected})")
            
            results.append({
                'expression': expr,
                'predicted': result,
                'expected': expected,
                'correct': correct
            })
            
        except Exception as e:
            print(f"✗ {expr} - Error: {e}")
            logging.error(f"Router test failed for {expr}: {e}", exc_info=True)
            results.append({
                'expression': expr,
                'predicted': None,
                'expected': expected,
                'correct': False
            })
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) * 100
    
    print(f"\n{'='*80}")
    print(f"Router Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train and test FoNE expert models"
    )
    
    # Training options
    parser.add_argument('--train_all', action='store_true',
                       help='Train all expert models')
    parser.add_argument('--operation', type=str, choices=list(EXPERT_CONFIGS.keys()),
                       help='Train specific operation expert')
    
    # Testing options
    parser.add_argument('--test_expert', type=str, choices=list(EXPERT_CONFIGS.keys()),
                       help='Test a specific trained expert')
    parser.add_argument('--test_router', action='store_true',
                       help='Test router with multi-operation expressions')
    
    # Configuration
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_train_samples', type=int, default=10000,
                       help='Number of training samples')
    
    args = parser.parse_args()
    
    # Prepare override arguments
    args_override = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_train_samples': args.num_train_samples
    }
    
    # Execute requested action
    if args.train_all:
        train_all_experts()
        
    elif args.operation:
        train_expert(args.operation, args_override)
        
    elif args.test_expert:
        # Test expressions for each operation
        test_sets = {
            'addition': ['100+50', '25+75', '123+456', '999+1'],
            'subtraction': ['100-50', '75-25', '456-123', '1000-1'],
            'multiplication': ['10*5', '12*3', '25*4', '100*2'],
            'division': ['100/2', '75/3', '144/12', '1000/10']
        }
        test_single_expert(args.test_expert, test_sets[args.test_expert])
        
    elif args.test_router:
        test_router_multi_operation()
        
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Train all experts:    python train_experts.py --train_all")
        print("  2. Test single expert:   python train_experts.py --test_expert addition")
        print("  3. Test router:          python train_experts.py --test_router")


if __name__ == "__main__":
    main()