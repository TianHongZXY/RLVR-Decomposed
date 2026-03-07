import os
import torch
import glob
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

def clean_tensor(tensor):
    """
    Force convert DTensor or other FSDP-wrapped tensors to clean CPU tensors.
    """
    # Check if it has to_local method (characteristic of DTensor)
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    
    # Detach from computation graph and convert to regular CPU tensor
    return tensor.detach().clone().cpu()

def merge_fsdp_checkpoints(src_dir, output_dir):
    print(f"Starting merge from: {src_dir}")
    print(f"Output directory: {output_dir}")
    
    # 1. Automatically find config file
    config_path = os.path.join(src_dir, "huggingface")
    if not os.path.exists(os.path.join(config_path, "config.json")):
        config_path = src_dir
        if not os.path.exists(os.path.join(config_path, "config.json")):
            raise ValueError(f"Cannot find config.json in {src_dir} or {src_dir}/huggingface")
    
    print(f"Loading config from: {config_path}")
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    
    # 2. Initialize target model structure
    # Use meta device to avoid immediate memory allocation; load_state_dict will handle it later
    print("Initializing empty model structure...")
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception:
        print("Meta device initialization failed, falling back to CPU...")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Get target shapes
    target_shapes = {k: v.shape for k, v in model.state_dict().items()}
    
    # 3. Load all shards
    print("Loading FSDP shards...")
    shard_files = glob.glob(os.path.join(src_dir, "model_world_size_*.pt"))
    if not shard_files:
        raise ValueError("No matching 'model_world_size_*.pt' files found!")
    
    # Sort by rank
    try:
        shard_files.sort(key=lambda x: int(x.split("rank_")[-1].split(".")[0]))
    except Exception:
        shard_files.sort()
    
    print(f"Found {len(shard_files)} shards. Loading into memory...")
    
    # Load shards; note: map_location='cpu' may not be sufficient to remove DTensor attributes
    raw_shards = [torch.load(f, map_location="cpu") for f in shard_files]
    
    # 4. Merge weights
    print("Merging weights and scrubbing DTensors...")
    new_state_dict = {}
    
    for key, target_shape in target_shapes.items():
        # Find corresponding key
        found_key = None
        # Possible key variants when FSDP saves
        possible_keys = [key, f"_fsdp_wrapped_module.{key}", f"model.{key}"]
        
        # Search in the first shard
        for k in possible_keys:
            if k in raw_shards[0]:
                found_key = k
                break
        
        if not found_key:
            # Some buffers (e.g. causal mask) don't need to be loaded, skip
            continue
            
        # Extract slices and "clean" them (strip DTensor)
        slices = [clean_tensor(s[found_key]) for s in raw_shards]
        
        # --- Merge logic ---
        # Case A: Shapes are identical (Replicated)
        if slices[0].shape == target_shape:
            merged_tensor = slices[0]
            
        # Case B: Dimension 0 split (Row Parallel)
        elif sum(s.shape[0] for s in slices) >= target_shape[0] and slices[0].shape[1:] == target_shape[1:]:
            merged_tensor = torch.cat(slices, dim=0)
            if merged_tensor.shape[0] > target_shape[0]:
                merged_tensor = merged_tensor[:target_shape[0]]
                
        # Case C: Dimension 1 split (Column Parallel)
        elif len(target_shape) > 1 and sum(s.shape[1] for s in slices) >= target_shape[1] and slices[0].shape[0] == target_shape[0]:
            merged_tensor = torch.cat(slices, dim=1)
            if merged_tensor.shape[1] > target_shape[1]:
                merged_tensor = merged_tensor[:, :target_shape[1]]
        
        # Case D: Only 1D vector is split (bias, etc.)
        elif len(target_shape) == 1 and sum(s.shape[0] for s in slices) >= target_shape[0]:
             merged_tensor = torch.cat(slices, dim=0)
             if merged_tensor.shape[0] > target_shape[0]:
                merged_tensor = merged_tensor[:target_shape[0]]

        else:
            print(f"Error merging {key}: Target {target_shape}, Slices {[s.shape for s in slices]}")
            raise RuntimeError(f"Shape mismatch for {key}")
            
        new_state_dict[key] = merged_tensor

    # Release raw shards to save memory
    del raw_shards
    gc.collect()

    # 5. Save
    print(f"Saving merged model to {output_dir}...")
    
    # meta device model cannot load_state_dict directly (assign=True may not support all operations)
    # Re-instantiate a pure CPU model for reliability
    del model
    print("Instantiating CPU model for saving...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Load cleaned weights
    model.load_state_dict(new_state_dict)
    
    # Save in HuggingFace format (Safetensors)
    model.save_pretrained(output_dir)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")
    except Exception:
        print("Warning: Could not save tokenizer. Copy it manually if needed.")

    print("Done! Conversion successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    merge_fsdp_checkpoints(args.src, args.out)
