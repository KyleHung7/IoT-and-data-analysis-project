import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from LSTM.evaluate_model import load_model
from LSTM.train_model import build_sequences_from_dataframe
from LSTM.config import SEQUENCE_LENGTH
import os

def validate_model(model_path, data_path):
    print(f"Validating model: {model_path}")
    print(f"Data: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows.")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, normalizer_params, model_config = load_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Build sequences
    # Use the loaded normalizer params if available, otherwise fit (fallback)
    fit_norm = False
    if normalizer_params is None:
        print("Warning: No normalizer params found in checkpoint. Fitting new scaler (may degrade performance if distribution differs).")
        fit_norm = True
        
    X, y, _ = build_sequences_from_dataframe(
        df, 
        sequence_length=SEQUENCE_LENGTH, 
        normalize=True, 
        fit_normalizer=fit_norm,
        normalizer_params=normalizer_params
    )
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    model.eval()
    
    # Predict
    with torch.no_grad():
        outputs = model(X_tensor.to(device))
        predictions = (outputs > 0.5).float().cpu().numpy()
        
    # Metrics
    y_true = y_tensor.cpu().numpy()
    
    print("\n" + "="*30)
    print("VALIDATION RESULTS")
    print("="*30)
    print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))
    print("\nClassification Report:")
    print(classification_report(y_true, predictions, target_names=['GO', 'STOP']))

if __name__ == "__main__":
    finetuned_model = os.path.join("LSTM", "models", "finetuned", "best_model.pt")
    data_csv = "ground_truth_processed.csv"
    
    if os.path.exists(finetuned_model):
        validate_model(finetuned_model, data_csv)
    else:
        print(f"Model file not found: {finetuned_model}")
