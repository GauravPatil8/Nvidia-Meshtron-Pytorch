import torch
import torch.nn as nn
from torchinfo import summary
from src.components.model import get_model
from src.config import ConfigurationManager

# Initialize model
model = get_model(model_params=ConfigurationManager.model_params())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Meshtron Model Summary")
print("*" * 100)

# Define input sizes with proper structure for torchinfo
# Based on your comment: data, conditioning_data, face_count, quad_ratio, mask
input_sizes = [
    (10, 10),           # data
    (10, 10, 6),        # conditioning_data  
    (10,),              # face_count
    (10,),              # quad_ratio
    (10, 1, 10, 10)     # mask (main hourglass input)
]

# try:
#     # Using torchinfo with proper parameters
#     summary(
#         model, 
#         input_size=input_sizes,
#         verbose=1,
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
#     )
    
# except Exception as e:
#     print(f"Error with torchinfo summary: {e}")
#     print("\nTrying alternative approach...")
    
    # Alternative: Manual analysis
try:
        # Create dummy inputs
        dummy_inputs = [
            torch.randn(10, 10).to(device),           # data
            torch.randn(10, 10, 6).to(device),        # conditioning_data
            torch.randn(10).to(device),               # face_count
            torch.randn(10).to(device),               # quad_ratio
            torch.randn(10, 1, 10, 10).to(device)     # mask
        ]
        
        print("Input shapes:")
        input_names = ["data", "conditioning_data", "face_count", "quad_ratio", "mask"]
        for i, (name, inp) in enumerate(zip(input_names, dummy_inputs)):
            print(f"  {name}: {inp.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(*dummy_inputs)
            
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print("\n" + "=" * 80)
        print("MODEL STATISTICS")
        print("=" * 80)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")
        
        # Output information
        if isinstance(outputs, torch.Tensor):
            print(f"\nOutput shape: {outputs.shape}")
        elif isinstance(outputs, (list, tuple)):
            print(f"\nNumber of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                if hasattr(out, 'shape'):
                    print(f"  Output {i+1}: {out.shape}")
                else:
                    print(f"  Output {i+1}: {type(out)}")
        else:
            print(f"\nOutput type: {type(outputs)}")
            if hasattr(outputs, 'shape'):
                print(f"Output shape: {outputs.shape}")
        
        print("\n" + "=" * 80)
        print("MEMORY USAGE ESTIMATION")
        print("=" * 80)
        
        # Calculate input memory usage
        input_memory = 0
        for inp in dummy_inputs:
            input_memory += inp.numel() * 4  # 4 bytes per float32
        
        print(f"Input memory (MB): {input_memory / (1024**2):.2f}")
        
        # Estimate forward pass memory (rough approximation)
        # This is a very rough estimate
        forward_memory = total_params * 4 * 2  # params + gradients approximation
        print(f"Estimated forward pass memory (MB): {forward_memory / (1024**2):.2f}")
        
        print("\n" + "=" * 80)
        print("MODEL ARCHITECTURE OVERVIEW")
        print("=" * 80)
        
        # Print model structure
        print(model)
        
except Exception as e2:
        print(f"Error with manual analysis: {e2}")
        
        # Fallback: Just print model info
        print("\nFallback - Basic model information:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Model: {type(model).__name__}")
        print("\nModel structure:")
        print(model)

print("\n" + "*" * 100)
print("Summary completed!")