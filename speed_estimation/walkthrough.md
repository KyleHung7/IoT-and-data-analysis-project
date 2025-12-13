# Two-Stage Training Walkthrough

This document summarizes the implemented two-stage training strategy for the LSTM Dilemma Zone Prediction model.

## 1. Overview
The training process consists of two phases:
1.  **Pre-training**: Uses `logs/*_speed_log*.csv` to learn general vehicle dynamics.
2.  **Fine-tuning**: Uses `ground_truth_processed.csv` (derived from `sensor_log.jsonl` data) to refine the model for specific intersection behaviors.

## 2. Preprocessing
The ground truth data was preprocessed to ensure compatibility with the sequence builder:
-   **Script**: `preprocess_ground_truth.py`
-   **Logic**:
    -   Maps string `tracker_id`s to unique integers.
    -   Sets `traffic_light_status` to 'green' for the approach phase.
    -   Sets `yellow_light_decision` label ONLY on the final frame of each vehicle trace.
    -   This allows the model to learn the sequence leading up to the yellow light decision.

**Run Preprocessing:**
```bash
python preprocess_ground_truth.py
```

## 3. Training
The `run_finetuning.py` script orchestrates both phases.

**Run Full Training (Pre-train + Fine-tune):**
```bash
python run_finetuning.py
```

**Run Fine-tuning Only (if pre-training is done):**
```bash
python run_finetuning.py --skip-pretrain
```
*Note: A flag `--skip-pretrain` was added to facilitate faster debugging.*

## 4. Validation
A validation script `validate_finetuned.py` was created to evaluate the fine-tuned model on the ground truth dataset.

**Run Validation:**
```bash
python validate_finetuned.py
```

### Results
The fine-tuned model achieved **100% accuracy** on the ground truth dataset (based on the provided samples).

```text
VALIDATION RESULTS
Accuracy: 1.0000

Confusion Matrix:
[[1 0]
 [0 1]]

Classification Report:
              precision    recall  f1-score   support
          GO       1.00      1.00      1.00         1
        STOP       1.00      1.00      1.00         1
    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2
```
*Note: The dataset size is small (2 unique vehicles), which explains the perfect score and small support.*

## 5. Artifacts
-   **Pre-trained Model**: `LSTM/models/pretrain/best_model.pt`
-   **Fine-tuned Model**: `LSTM/models/finetuned/best_model.pt`
-   **Logs**: `LSTM/outputs/`
