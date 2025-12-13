import os
import sys
import argparse
from pathlib import Path
from LSTM.train_model import train

def main():
    parser = argparse.ArgumentParser(description='Run two-stage training: Pre-training + Fine-tuning')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Directory containing log CSVs for pre-training')
    parser.add_argument('--ground_truth_csv', type=str, default='ground_truth_processed.csv', help='Path to processed ground truth CSV for fine-tuning')
    parser.add_argument('--base_output_dir', type=str, default='LSTM/models', help='Base directory for model outputs')
    
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip pre-training phase')
    
    args = parser.parse_args()
    
    # Paths
    pretrain_output_dir = os.path.join(args.base_output_dir, 'pretrain')
    finetune_output_dir = os.path.join(args.base_output_dir, 'finetuned')
    
    # --- Phase 1: Pre-training ---
    if not args.skip_pretrain:
        print("\n" + "="*50)
        print("PHASE 1: Pre-training on Log Data")
        print("="*50)
        print(f"Data Source: Directory '{args.logs_dir}'")
        print(f"Output: {pretrain_output_dir}")
        
        try:
            train(
                csv_directory=args.logs_dir,
                csv_pattern="*_speed_log*.csv",
                output_dir=pretrain_output_dir,
                num_epochs=50,
                learning_rate=0.001
            )
        except Exception as e:
            print(f"Error during pre-training: {e}")
            if not os.path.exists(os.path.join(pretrain_output_dir, 'best_model.pt')):
                print("Pre-training failed to produce a model. Aborting.")
                sys.exit(1)
        
        print(f"\nPre-training complete. Best model saved to: {os.path.join(pretrain_output_dir, 'best_model.pt')}")
    else:
        print("Skipping pre-training phase...")

    # Check for pre-trained model
    pretrain_model_path = os.path.join(pretrain_output_dir, 'best_model.pt')
    if not os.path.exists(pretrain_model_path):
        print(f"Pre-trained model not found at {pretrain_model_path}. Aborting.")
        sys.exit(1)

    # --- Phase 2: Fine-tuning ---
    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning on Ground Truth Data")
    print("="*50)
    print(f"Data Source: File '{args.ground_truth_csv}'")
    print(f"Resuming from: {pretrain_model_path}")
    print(f"Output: {os.path.abspath(finetune_output_dir)}")
    
    try:
        train(
            csv_path=args.ground_truth_csv,
            output_dir=finetune_output_dir,
            resume_from=pretrain_model_path,
            num_epochs=100,
            learning_rate=0.0001
        )
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Final Fine-tuned Model: {os.path.join(finetune_output_dir, 'best_model.pt')}")

if __name__ == '__main__':
    main()
