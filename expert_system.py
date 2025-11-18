"""
FoNE Expert Model System with Router
=====================================
This implements a multi-expert system where separate models are trained
for each arithmetic operation and a router orchestrates them.
"""

import torch
import torch.nn as nn
import os
import re
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

# ============================================================================
# PART 1: EXPERT MODEL MANAGEMENT
# ============================================================================

class ExpertModelManager:
    """Manages saving/loading of expert models trained on specific operations."""
    
    def __init__(self, base_save_dir="./saved_experts"):
        self.base_save_dir = base_save_dir
        os.makedirs(base_save_dir, exist_ok=True)
        
    def save_expert(self, model, fne, operation: str, metadata: dict = None):
        """
        Save a trained expert model and its FNE embedding.
        
        Args:
            model: The base language model
            fne: The FNE embedding module
            operation: Operation name (e.g., 'addition', 'subtraction')
            metadata: Additional info like accuracy, training params
        """
        save_path = os.path.join(self.base_save_dir, operation)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and FNE weights
        torch.save({
            'model_state_dict': model.state_dict(),
            'fne_state_dict': fne.state_dict(),
            'model_config': model.config.to_dict(),
            'fne_config': {
                'embedding_dim': fne.embedding_dim,
                'int_digit_len': fne.max_num_digits - len(fne.powers_of_ten) + fne.max_num_digits,
                'frac_digit_len': 0,  # Adjust based on your needs
                'period_base_list': [float(p) for p in fne.period_list[:10]]  # Sample
            },
            'metadata': metadata or {}
        }, os.path.join(save_path, 'checkpoint.pt'))
        
        logging.info(f"Saved {operation} expert to {save_path}")
        
    def load_expert(self, model, fne, operation: str, device='cuda'):
        """
        Load a trained expert model.
        
        Returns:
            Tuple of (loaded_model, loaded_fne, metadata)
        """
        checkpoint_path = os.path.join(self.base_save_dir, operation, 'checkpoint.pt')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No expert found for {operation} at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load state dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        fne.load_state_dict(checkpoint['fne_state_dict'])
        
        model.eval()
        fne.eval()
        
        logging.info(f"Loaded {operation} expert from {checkpoint_path}")
        logging.info(f"Metadata: {checkpoint['metadata']}")
        
        return model, fne, checkpoint['metadata']


# ============================================================================
# PART 2: EXPRESSION PARSER
# ============================================================================

@dataclass
class SubProblem:
    """Represents a single arithmetic operation."""
    left_operand: float
    operator: str
    right_operand: float
    position: int  # Position in original expression
    

class ExpressionParser:
    """Parses arithmetic expressions into subproblems respecting PEMDAS."""
    
    # Operator precedence (higher = evaluated first)
    PRECEDENCE = {
        '*': 2,
        '/': 2,
        '+': 1,
        '-': 1
    }
    
    @staticmethod
    def tokenize(expression: str) -> List[str]:
        """Convert expression string into tokens."""
        # Remove spaces
        expression = expression.replace(' ', '')
        
        # Split by operators while keeping them
        tokens = re.findall(r'\d+\.?\d*|[+\-*/()]', expression)
        return tokens
    
    @staticmethod
    def simple_left_to_right(expression: str) -> List[SubProblem]:
        """
        Parse expression left-to-right (NO PEMDAS).
        Example: "1+2-3*4" becomes [(1,+,2), (result,-,3), (result,*,4)]
        """
        tokens = ExpressionParser.tokenize(expression)
        
        subproblems = []
        
        # Filter out only numbers and operators (no parentheses for now)
        filtered = [t for t in tokens if t not in '()']
        
        i = 0
        while i < len(filtered):
            if i == 0:
                # First subproblem
                left = float(filtered[i])
                op = filtered[i+1]
                right = float(filtered[i+2])
                subproblems.append(SubProblem(left, op, right, 0))
                i += 2
            else:
                # Subsequent subproblems use 'RESULT' placeholder
                left = 'RESULT'
                op = filtered[i]
                right = float(filtered[i+1])
                subproblems.append(SubProblem(left, op, right, len(subproblems)))
                i += 2
                
            if i >= len(filtered) - 1:
                break
                
        return subproblems
    
    @staticmethod
    def with_pemdas(expression: str) -> List[SubProblem]:
        """
        Parse expression respecting PEMDAS order.
        Returns subproblems in evaluation order.
        """
        # TODO: Implement full PEMDAS parser
        # For now, will only be using simple left-to-right arithmetic expressions (i.e 10*5-20=)
        # Require building an expression tree
        raise NotImplementedError("PEMDAS parsing coming in Phase 2")


# ============================================================================
# PART 3: EXPERT ROUTER
# ============================================================================

class ExpertRouter:
    """Routes arithmetic subproblems to appropriate expert models."""
    
    OPERATION_MAP = {
        '+': 'addition',
        '-': 'subtraction',
        '*': 'multiplication',
        '/': 'division'
    }
    
    def __init__(self, expert_manager: ExpertModelManager, 
                 base_model, base_fne, device='cuda'):
        """
        Initialize router with expert models.
        
        Args:
            expert_manager: Manager for loading experts
            base_model: Base model architecture (for loading)
            base_fne: Base FNE architecture (for loading)
            device: Computation device
        """
        self.expert_manager = expert_manager
        self.device = device
        self.experts = {}
        
        # Store base architectures for cloning
        self.base_model = base_model
        self.base_fne = base_fne
        
    def load_expert(self, operation: str):
        """Load a specific expert if not already loaded."""
        if operation not in self.experts:
            # Create fresh model and FNE instances
            import copy
            model_copy = copy.deepcopy(self.base_model)
            fne_copy = copy.deepcopy(self.base_fne)
            
            # Load trained weights
            model, fne, metadata = self.expert_manager.load_expert(
                model_copy, fne_copy, operation, self.device
            )
            
            self.experts[operation] = {
                'model': model,
                'fne': fne,
                'metadata': metadata
            }
            logging.info(f"Loaded {operation} expert")
    
    def solve_subproblem(self, subproblem: SubProblem, 
                         tokenizer, int_digit_len=10, frac_digit_len=0) -> float:
        """
        Solve a single subproblem using the appropriate expert.
        
        Args:
            subproblem: The arithmetic operation to solve
            tokenizer: Tokenizer for creating inputs
            
        Returns:
            Numerical result
        """
        operation_name = self.OPERATION_MAP[subproblem.operator]
        
        # Load expert if needed
        if operation_name not in self.experts:
            self.load_expert(operation_name)
        
        expert = self.experts[operation_name]
        model = expert['model']
        fne = expert['fne']
        
        # Format input: "NUM op NUM ="
        left = subproblem.left_operand
        right = subproblem.right_operand
        
        # Create input string
        input_text = f" [NUM]  {subproblem.operator}  [NUM]  ="
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Create scatter tensor with actual numbers
        num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
        scatter_tensor = torch.zeros(input_ids.shape[1], dtype=torch.float64, device=self.device)
        
        num_positions = (input_ids[0] == num_token_id).nonzero(as_tuple=True)[0]
        if len(num_positions) >= 2:
            scatter_tensor[num_positions[0]] = left
            scatter_tensor[num_positions[1]] = right
        
        scatter_tensor = scatter_tensor.unsqueeze(0)  # Add batch dimension
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings
        with torch.no_grad():
            from train.utils import get_regular_embeddings
            regular_embeddings = get_regular_embeddings(model, input_ids)
            fourier_embeddings = fne(scatter_tensor)
            fourier_embeddings = fourier_embeddings.to(dtype=regular_embeddings.dtype)
            input_embeddings = regular_embeddings + fourier_embeddings
            
            # Forward pass
            outputs = model(inputs_embeds=input_embeddings, 
                          attention_mask=attention_mask, 
                          output_hidden_states=True)
            
            before_decoder = outputs.hidden_states[-1]
            last_token_hidden = before_decoder[:, -1, :]  # Last token
            
            # Predict result
            result = fne.fourier_compute_prediction(
                last_token_hidden, int_digit_len, frac_digit_len
            )
        
        return result.item()
    
    def solve_expression(self, expression: str, tokenizer, 
                        use_pemdas=False, int_digit_len=10, frac_digit_len=0) -> float:
        """
        Solve a complete arithmetic expression by routing to experts.
        
        Args:
            expression: Arithmetic expression string (e.g., "12+5-3")
            tokenizer: Tokenizer for model inputs
            use_pemdas: Whether to respect PEMDAS order
            
        Returns:
            Final numerical result
        """
        # Parse expression into subproblems
        if use_pemdas:
            subproblems = ExpressionParser.with_pemdas(expression)
        else:
            subproblems = ExpressionParser.simple_left_to_right(expression)
        
        logging.info(f"Expression '{expression}' parsed into {len(subproblems)} subproblems")
        
        # Solve each subproblem sequentially
        result = None
        for i, subproblem in enumerate(subproblems):
            # Replace 'RESULT' placeholder with actual previous result
            if subproblem.left_operand == 'RESULT':
                subproblem.left_operand = result
            
            logging.info(f"Subproblem {i+1}: {subproblem.left_operand} {subproblem.operator} {subproblem.right_operand}")
            
            result = self.solve_subproblem(subproblem, tokenizer, 
                                          int_digit_len, frac_digit_len)
            
            logging.info(f"  → Result: {result}")
        
        return result


# ============================================================================
# PART 4: TRAINING SCRIPT FOR EXPERTS
# ============================================================================

def train_single_expert(operation: str, dataset_name: str, args, 
                       model, tokenizer, fne, device):
    """
    Train a single expert model on one operation.
    
    This wraps your existing training pipeline.
    """
    from train.train_pipeline import create_dataloader_and_train
    
    # Override args for this specific operation
    args.dataset = dataset_name
    args.name = f'expert_{operation}'
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Training {operation.upper()} Expert")
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"{'='*80}\n")
    
    # Train using existing pipeline
    create_dataloader_and_train(args, model, tokenizer, device)
    
    # Save the trained expert
    manager = ExpertModelManager()
    manager.save_expert(model, fne, operation, metadata={
        'dataset': dataset_name,
        'accuracy': 'check_logs',  # You can extract from training
        'epochs': args.epochs
    })


# ============================================================================
# PART 5: USAGE EXAMPLES
# ============================================================================

def example_train_all_experts():
    """Example: Train all 4 expert models."""
    import argparse
    from utils.model_utils import load_model_and_tokenizer
    from number_encoders.FNE import FNE
    from utils.logger_utils import get_embedding_dim
    
    # Setup args (similar to your main.py)
    args = argparse.Namespace(
        batch_size=32,
        epochs=5,
        int_digit_len=10,
        frac_digit_len=0,
        len_gen_size=0,
        lr=5e-4,
        model='Qwen/Qwen2.5-7B-Instruct',
        train_from_scratch=True,
        model_size_level=4,
        num_train_samples=10000,
        seed=42,
        method='fne',
        period_base_list=[10.0],
        clip=True,
        add_linear=True,
        scheduler_name='cosine',
        use_digit_wise_tokenizer=False,
        num_test_samples=None
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset mapping for each operation
    expert_datasets = {
        'addition': 'Onlydrinkwater/int_addition',
        'subtraction': 'Onlydrinkwater/int_subtract',  # You'll need these
        'multiplication': 'Onlydrinkwater/int_multiplication',
        'division': 'Onlydrinkwater/int_division'
    }
    
    for operation, dataset in expert_datasets.items():
        # Load fresh model for each expert
        model, tokenizer = load_model_and_tokenizer(
            args.model, "./hg_cache", device,
            train_from_scratch=True, size_level=args.model_size_level
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
        
        # Train this expert
        train_single_expert(operation, dataset, args, model, tokenizer, fne, device)


def example_use_router():
    """Example: Use trained experts via router."""
    from utils.model_utils import load_model_and_tokenizer
    from number_encoders.FNE import FNE
    from utils.logger_utils import get_embedding_dim
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load base architectures (just for structure, weights will be replaced)
    model, tokenizer = load_model_and_tokenizer(
        'Qwen/Qwen2.5-7B-Instruct',
        "./hg_cache",
        device,
        train_from_scratch=True,
        size_level=4
    )
    
    embedding_dim = get_embedding_dim(model)
    fne = FNE(embedding_dim, int_digit_len=10, frac_digit_len=0,
              period_base_list=[10.0], device=device).to(device)
    
    # Initialize router
    manager = ExpertModelManager()
    router = ExpertRouter(manager, model, fne, device)
    
    # Test expressions
    test_cases = [
        "100+50-25",      # Left-to-right: 100+50=150, 150-25=125
        "10*2+5",         # Left-to-right: 10*2=20, 20+5=25
        "100-20-30",      # Left-to-right: 100-20=80, 80-30=50
    ]
    
    for expr in test_cases:
        result = router.solve_expression(expr, tokenizer, use_pemdas=False)
        print(f"{expr} = {result}")


if __name__ == "__main__":
    # Uncomment to train experts:
    example_train_all_experts()
    
    # Uncomment to test router:
    example_use_router()
    
    print("Expert Router System Ready!")
    print("\nNext steps:")
    print("1. Train experts: example_train_all_experts()")
    print("2. Test router: example_use_router()")