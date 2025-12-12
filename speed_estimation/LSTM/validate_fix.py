
import sys
import os
import torch

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LSTM.model_architecture import create_model, print_model_summary
from LSTM.config import MODEL_TYPE, FEATURE_DIM, SEQUENCE_LENGTH

def main():
    print("Attempting to instantiate model with new configuration...")
    try:
        model = create_model(
            model_type=MODEL_TYPE,
            input_dim=FEATURE_DIM,
            sequence_length=SEQUENCE_LENGTH
        )
        print("Model instantiated successfully!")
        
        print_model_summary(model)
        
        # Check if dropout is present
        print("\nChecking for dropout layers:")
        has_dropout = False
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                print(f"  Found dropout layer: {name}, p={module.p}")
                has_dropout = True
        
        if has_dropout:
            print("✓ Dropout layers verified")
        else:
            print("✗ WARNING: No dropout layers found!")
            
    except Exception as e:
        print(f"FAILED to instantiate model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
