# hierarchical/from_models.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM 
# from tqdm import tqdm # tqdm is not used in this file's functions directly

def get_gamma(MODEL_NAME, device): # Renamed in __init__ to get_output_embeddings_gamma
    # device_map logic as previously discussed for robustness
    map_location = device if isinstance(device, str) and (device == "cpu" or ":" in device) else "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                     torch_dtype=torch.float32,
                                                     device_map=map_location)
    except Exception as e:
        print(f"Retrying model load for {MODEL_NAME} (for get_gamma) with trust_remote_code=True. Original error: {e}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                     torch_dtype=torch.float32,
                                                     device_map=map_location,
                                                     trust_remote_code=True)
    
    if not hasattr(model, 'get_output_embeddings') or model.get_output_embeddings() is None:
        if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
            print(f"Warning (get_gamma): Model {MODEL_NAME} lacks 'get_output_embeddings'. Using 'get_input_embeddings' as proxy (e.g., for Gemma).")
            gamma_val = model.get_input_embeddings().weight.detach()
        else:
            raise AttributeError(f"Model {MODEL_NAME} in get_gamma doesn't have get_output_embeddings (or get_input_embeddings fallback).")
    else:
        gamma_val = model.get_output_embeddings().weight.detach()
        
    return gamma_val.to(device)

def get_g(MODEL_NAME, device): # Renamed in __init__ to get_transformed_g_and_covs
    # This calls the get_gamma in this file (which becomes hrc.get_output_embeddings_gamma)
    gamma_val = get_gamma(MODEL_NAME, device) 
    W, d = gamma_val.shape
    gamma_bar = torch.mean(gamma_val, dim=0, keepdim=True)
    centered_gamma = gamma_val - gamma_bar

    Cov_gamma = centered_gamma.T @ centered_gamma / W
    Cov_gamma_stable = Cov_gamma + torch.eye(d, device=device) * 1e-6 # Add epsilon for stability
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma_stable)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=1e-6) # Ensure positive before sqrt

    inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1.0/torch.sqrt(eigenvalues_clamped)) @ eigenvectors.T
    sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues_clamped)) @ eigenvectors.T
    g_transformed = centered_gamma @ inv_sqrt_Cov_gamma
    return g_transformed, inv_sqrt_Cov_gamma, sqrt_Cov_gamma

# --- MODIFIED get_vocab FUNCTION ---
def get_vocab(model_name_or_tokenizer_instance): # Renamed parameter, will be hrc.get_tokenizer_vocab via __init__
    """
    Returns a vocabulary dictionary (token -> id) and list of tokens (id -> token string).
    Accepts a model name string (to load tokenizer) or a pre-loaded tokenizer instance.
    """
    if isinstance(model_name_or_tokenizer_instance, str):
        # If it's a string, assume it's a model name and load the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_tokenizer_instance)
        except Exception as e:
            print(f"Retrying tokenizer load for '{model_name_or_tokenizer_instance}' with trust_remote_code=True. Error: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_tokenizer_instance, trust_remote_code=True)
            except Exception as e_trust:
                print(f"Fatal: Failed to load tokenizer for '{model_name_or_tokenizer_instance}'. Error: {e_trust}")
                raise
    # Check if it's a Hugging Face tokenizer object (basic check for common attributes)
    elif hasattr(model_name_or_tokenizer_instance, 'get_vocab') and \
         hasattr(model_name_or_tokenizer_instance, 'vocab_size') and \
         hasattr(model_name_or_tokenizer_instance, 'convert_ids_to_tokens'):
        tokenizer = model_name_or_tokenizer_instance
    else:
        raise TypeError(
            "get_vocab expects a model name string or a "
            f"valid Hugging Face Tokenizer instance, but got type {type(model_name_or_tokenizer_instance)}"
        )

    vocab_dict = tokenizer.get_vocab()
    # Ensure vocab_list is correctly sized and populated based on tokenizer.vocab_size
    vocab_list = [""] * tokenizer.vocab_size  # Initialize with empty strings
    for word, index in vocab_dict.items():
        if 0 <= index < tokenizer.vocab_size:  # Check index bounds against reported vocab_size
            vocab_list[index] = word
        # else: (Optional: print warning if an index from get_vocab() is outside vocab_size)
            # print(f"Warning (get_vocab): Token '{word}' has index {index} which is outside the tokenizer's vocab size of {tokenizer.vocab_size}. Skipping for vocab_list.")
            
    return vocab_dict, vocab_list
# --- END OF MODIFIED get_vocab FUNCTION ---

def compute_lambdas(texts, MODEL_NAME, device):
    # (Implementation as you provided)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except: # Basic fallback, consider specific exception handling
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    map_location = device if isinstance(device, str) and (device == "cpu" or ":" in device) else "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                     torch_dtype=torch.float32,
                                                     device_map=map_location)
    except: # Basic fallback
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                     torch_dtype=torch.float32,
                                                     device_map=map_location,
                                                     trust_remote_code=True)
    
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left" # Crucial for getting last non-pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print(f"Warning (compute_lambdas): tokenizer.pad_token is None. Using eos_token ('{tokenizer.eos_token}') as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Warning (compute_lambdas): tokenizer.pad_token and eos_token are None. Adding a new pad token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer)) # Important if new token added

    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        
        if tokenizer.padding_side == 'left':
            # Get the index of the last actual token for each sequence
            # (sum of non-pad tokens) - 1 gives the index of the last non-pad token
            sequence_lengths = (inputs.input_ids != tokenizer.pad_token_id).sum(dim=1) - 1
            # Gather the hidden states of these last tokens
            lambdas = outputs.hidden_states[-1][torch.arange(inputs.input_ids.shape[0], device=model.device), sequence_lengths, :]
        else: # Fallback if padding_side is not left (original code had assert for left)
            print("Warning (compute_lambdas): tokenizer.padding_side is not 'left'. Lambda extraction will use the very last token embedding, which might be padding.")
            lambdas = outputs.hidden_states[-1][:, -1, :]

    tokenizer.padding_side = original_padding_side # Restore original padding side
    return lambdas.to(device) # Ensure result is on the initially requested device
