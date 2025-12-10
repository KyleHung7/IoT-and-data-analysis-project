# Training Issues Analysis: Validation Loss < Training Loss

## Summary
**The labels are NOT switched.** The code is correct. However, there are two issues causing validation loss to be lower than training loss:

---

## Issue 1: Extremely Small and Completely Imbalanced Validation Set ⚠️

### Current Situation:
- **Training set**: 46 sequences
  - 40 STOP (87%)
  - 6 GO (13%)
  
- **Validation set**: 6 sequences
  - 6 STOP (100%)
  - 0 GO (0%)

### Why This Causes Lower Validation Loss:
1. **No GO examples in validation**: The model can achieve perfect validation loss by always predicting STOP
2. **Tiny validation set**: Only 6 sequences is too small for reliable metrics
3. **Class imbalance**: 100% STOP means the model doesn't need to learn GO behavior for validation

### Impact:
- Validation loss appears artificially low
- Model appears to be "overfitting" when it's actually just exploiting the imbalanced validation set
- Validation metrics are unreliable

---

## Issue 2: Dropout During Training (Expected Behavior) ✅

### Current Configuration:
- `LSTM_DROPOUT = 0.2` (20% dropout)
- Dropout is **active during training** (`model.train()`)
- Dropout is **disabled during validation** (`model.eval()`)

### Why This Can Cause Lower Validation Loss:
- **Training**: Model sees "harder" version of data (20% of neurons randomly disabled)
- **Validation**: Model sees "easier" version of data (all neurons active)
- This is **normal and expected** behavior - dropout is a regularization technique

### Impact:
- This is **not a bug** - it's how dropout works
- However, combined with Issue 1, it makes the problem worse

---

## What the Code Does (Verified Correct):

### Training Function (`train_epoch`):
```python
train_loss = train_epoch(model, train_loader, ...)  # Returns training loss
train_losses.append(train_loss)  # Stored in train_losses
```

### Validation Function (`validate`):
```python
val_loss = validate(model, val_loader, ...)  # Returns validation loss
val_losses.append(val_loss)  # Stored in val_losses
```

### Plotting Function:
```python
plt.plot(epochs, train_losses, label='Training Loss')  # Blue line
plt.plot(epochs, val_losses, label='Validation Loss')  # Red line
```

**✅ Labels are correct - no switching occurred**

---

## Recommendations:

### 1. Fix Validation Set Imbalance (CRITICAL)
- **Option A**: Use stratified split to ensure both classes in validation
  - Modify split logic to ensure at least 1-2 GO examples in validation
  - Use `train_test_split` with `stratify` parameter if possible

- **Option B**: Collect more data
  - Need more GO examples overall
  - Current dataset is heavily skewed (87% STOP)

- **Option C**: Adjust split ratio
  - Current: 80/20 split
  - With only 6 GO examples total, 20% = ~1 GO example
  - Consider 70/30 or even 60/40 to get more validation examples

### 2. Monitor Class Distribution
- Added validation checks in `train_model.py` to warn about:
  - Small validation sets (< 10 sequences)
  - Completely imbalanced validation sets (0% GO or 0% STOP)
  - These warnings will appear during training

### 3. Consider Alternative Metrics
- Since validation set is imbalanced, focus on:
  - **Precision/Recall** for STOP class
  - **F1 score** (balanced metric)
  - **AUC-ROC** (works well with imbalanced data)
  - Don't rely solely on loss

### 4. Dropout Consideration
- Current dropout (0.2) is reasonable
- If you want validation loss to be more comparable to training loss:
  - Reduce dropout to 0.1 or 0.0
  - But this may reduce regularization benefits

---

## Next Steps:

1. **Re-run training** with the updated code to see the new warnings
2. **Review the split strategy** - consider using stratified split
3. **Collect more GO examples** if possible
4. **Monitor class distribution** in both train and validation sets

---

## Code Changes Made:

1. **Added validation checks** in `train_model.py`:
   - Warns if validation set is too small (< 10 sequences)
   - Warns if validation set is completely imbalanced (0% GO or 0% STOP)
   - Prints label distribution for both train and validation sets

2. **Added explanatory note** in `dilemma_zone_generator.py`:
   - Training history plots now include a note explaining why validation loss might be lower
   - Helps users understand this is not necessarily overfitting

---

## Conclusion:

**The code is correct - labels are not switched.** The issue is:
1. **Validation set is too small and imbalanced** (main issue)
2. **Dropout makes training "harder"** (expected behavior)

The model is not truly overfitting - it's just exploiting the imbalanced validation set. Fix the validation set distribution to get reliable metrics.

