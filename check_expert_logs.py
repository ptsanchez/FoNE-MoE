#!/usr/bin/env python3
"""
Diagnostic tool to debug dtype issues in FoNE expert loading.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model_utils import load_model_and_tokenizer
from utils.logger_utils import get_embedding_dim
from number_encoders.FNE import FNE
from expert_system import ExpertModelManager

def diagnose_expert(operation='addition'):
    """Diagnose dtype issues when loading an expert."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("DTYPE DIAGNOSTIC TOOL")
    print("="*80)
    
    # Load base model
    print("\n1. Loading base model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name='Qwen/Qwen2.5-7B-Instruct',
        cache_dir="./hg_cache",
        device=device,
        train_from_scratch=True,
        size_level=4
    )
    
    embedding_dim = get_embedding_dim(model)
    
    print(f"   ✓ Model loaded")
    print(f"   - Embedding dim: {embedding_dim}")
    print(f"   - Model dtype: {next(model.parameters()).dtype}")
    
    # Initialize FNE
    print("\n2. Initializing FNE...")
    fne = FNE(
        embedding_dim,
        int_digit_len=10,
        frac_digit_len=0,
        period_base_list=[10.0],
        add_linear=True,
        device=device
    ).to(device)
    
    print(f"   ✓ FNE initialized")
    print(f"   - FNE linear weight dtype: {fne.linear.weight.dtype}")
    print(f"   - FNE layer_norm weight dtype: {fne.layer_norm.weight.dtype}")
    
    # Load expert weights
    print(f"\n3. Loading {operation} expert weights...")
    manager = ExpertModelManager()
    
    try:
        model, fne, metadata = manager.load_expert(model, fne, operation, device)
        print(f"   ✓ Expert loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load expert: {e}")
        return
    
    print(f"   - Model dtype after load: {next(model.parameters()).dtype}")
    print(f"   - FNE linear weight dtype: {fne.linear.weight.dtype}")
    print(f"   - FNE layer_norm weight dtype: {fne.layer_norm.weight.dtype}")
    
    # Test a simple forward pass
    print("\n4. Testing forward pass...")
    
    # Create test input
    input_text = " [NUM]  +  [NUM]  ="
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
    scatter_tensor = torch.zeros(input_ids.shape[1], dtype=torch.float64, device=device)
    num_positions = (input_ids[0] == num_token_id).nonzero(as_tuple=True)[0]
    
    if len(num_positions) >= 2:
        scatter_tensor[num_positions[0]] = 100.0
        scatter_tensor[num_positions[1]] = 50.0
    
    scatter_tensor = scatter_tensor.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    
    # Create last_token_mask like in training
    last_token_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    last_token_mask[0, -1] = 1.0  # Mark last position
    
    print(f"   - Input IDs dtype: {input_ids.dtype}")
    print(f"   - Scatter tensor dtype: {scatter_tensor.dtype}")
    print(f"   - Attention mask dtype: {attention_mask.dtype}")
    print(f"   - Last token mask dtype: {last_token_mask.dtype}")
    
    try:
        with torch.no_grad():
            # Get embeddings
            from train.utils import get_regular_embeddings
            regular_embeddings = get_regular_embeddings(model, input_ids)
            print(f"   - Regular embeddings dtype: {regular_embeddings.dtype}")
            
            # FNE forward pass
            fourier_embeddings = fne(scatter_tensor)
            print(f"   - Fourier embeddings dtype (raw): {fourier_embeddings.dtype}")
            
            # Convert to model dtype
            model_dtype = regular_embeddings.dtype
            fourier_embeddings = fourier_embeddings.to(dtype=model_dtype)
            print(f"   - Fourier embeddings dtype (converted): {fourier_embeddings.dtype}")
            
            # Combine
            input_embeddings = regular_embeddings + fourier_embeddings
            print(f"   - Combined embeddings dtype: {input_embeddings.dtype}")
            
            # Model forward
            attention_mask = attention_mask.long()
            outputs = model(inputs_embeds=input_embeddings,
                          attention_mask=attention_mask,
                          output_hidden_states=True)
            
            before_decoder = outputs.hidden_states[-1]
            print(f"   - Hidden states dtype: {before_decoder.dtype}")
            
            last_token_hidden = before_decoder[:, -1, :]
            print(f"   - Last token hidden dtype: {last_token_hidden.dtype}")
            
            # Convert to float32 for FNE prediction
            last_token_hidden_float = last_token_hidden.float()
            print(f"   - Last token hidden (float32): {last_token_hidden_float.dtype}")
            
            # FNE prediction
            result = fne.fourier_compute_prediction(last_token_hidden_float, 10, 0)
            print(f"   ✓ Prediction successful: {result.item():.2f}")
            print(f"   - Expected: 150.0")
            
    except Exception as e:
        print(f"   ✗ Forward pass failed!")
        print(f"   - Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    diagnose_expert('addition')