# SHAP Analysis Interpretation Guide

## Overview
SHAP (SHapley Additive exPlanations) values explain how each feature contributes to your model's predictions. For the dilemma zone prediction model, SHAP values tell us which factors (speed, distance, traffic conditions) most influence whether a vehicle will STOP or GO at a yellow light.

---

## 1. SHAP Summary Bar Plot (`shap_summary_bar.png`)

### What it shows:
- **Mean absolute SHAP value** for each feature (aggregated across all timesteps)
- Features ranked from most important (top) to least important (bottom)

### Key Findings from Your Results:

**Most Important Features:**
1. **`distance_to_stop_line`** (0.012) - **CRITICAL**
   - This is by far the most important feature
   - The distance to the stop line has the strongest impact on STOP/GO decisions
   - **Interpretation**: Vehicles closer to the stop line are more likely to STOP

2. **`distance_to_front_vehicle`** (0.007) - **IMPORTANT**
   - Second most important feature
   - **Interpretation**: When there's a vehicle ahead, drivers are more cautious

3. **`traffic_density`** (0.003) - **MODERATE**
   - Has some influence but less than distance features
   - **Interpretation**: Higher traffic density affects decision-making

4. **`class_id`** (0.001) - **MINIMAL**
   - Vehicle type (car, truck, motorcycle) has very little impact
   - **Interpretation**: Vehicle type doesn't significantly affect STOP/GO decisions

5. **`ttc` (Time To Collision)** (~0.000) - **NEGLIGIBLE**
   - Almost no impact on predictions
   - **Interpretation**: TTC is not being used effectively by the model

6. **`speed_ms`** (~0.000) - **NEGLIGIBLE**
   - Almost no impact on predictions
   - **Interpretation**: Current speed is not a strong predictor in this model

### Business Insight:
- **Distance-based features dominate**: The model primarily relies on spatial information (how far from stop line, how far from front vehicle)
- **Speed and TTC are underutilized**: These features should theoretically be important but aren't contributing much
- **Vehicle type doesn't matter**: All vehicle types follow similar decision patterns

---

## 2. SHAP Summary Dot Plot (`shap_summary_dot.png`)

### What it shows:
- **Distribution of SHAP values** for each feature across all data points
- **Color coding**: Blue = low feature value, Red = high feature value
- **Position**: Left (negative SHAP) = pushes prediction toward STOP, Right (positive SHAP) = pushes prediction toward GO

### Key Findings from Your Results:

**`distance_to_stop_line`:**
- **Red dots (high distance)** → Positive SHAP values (right side) → **GO decision**
- **Blue dots (low distance)** → Near zero or negative SHAP → **STOP decision**
- **Interpretation**: 
  - Far from stop line = GO (have time to cross)
  - Close to stop line = STOP (too risky to proceed)

**`distance_to_front_vehicle`:**
- **Blue dots (low distance = close to front vehicle)** → Positive SHAP (right) → **GO decision**
- **Red dots (high distance = far from front vehicle)** → Negative SHAP (left) → **STOP decision**
- **Interpretation**: 
  - Close to front vehicle = GO (following closely, likely to follow their lead)
  - Far from front vehicle = STOP (more independent decision, more cautious)

**`traffic_density`:**
- **Red dots (high density)** → Positive SHAP → **GO decision**
- **Blue dots (low density)** → Negative SHAP → **STOP decision**
- **Interpretation**: 
  - High traffic = GO (pressure to keep moving)
  - Low traffic = STOP (more conservative)

**`ttc` and `speed_ms`:**
- Both clustered around zero → **Minimal impact**
- **Interpretation**: These features are not contributing meaningfully to predictions

### Business Insight:
- **Distance to stop line is the primary decision factor**: Clear separation between high/low values
- **Front vehicle distance has counterintuitive pattern**: May indicate following behavior
- **Speed and TTC are not being leveraged**: Model is missing important safety signals

---

## 3. SHAP Temporal Bar Plot (`shap_summary_temporal_bar.png`)

### What it shows:
- **Mean absolute SHAP value** for each feature **at each timestep** (t0 to t7)
- Shows how feature importance changes over time within the sequence
- Features are shown as `feature_name_tX` where X is the timestep

### Key Findings from Your Results:

**`distance_to_stop_line` across timesteps:**
- **t3, t4, t2, t5, t1, t6** (all ~0.02) - **HIGHEST IMPORTANCE**
- **t7, t0** (~0.01) - **MODERATE IMPORTANCE**
- **Pattern**: Middle timesteps (t1-t6) are most important
- **Interpretation**: 
  - Recent history (t1-t6) is critical for decision-making
  - Very recent (t7) and very old (t0) are less important
  - The model focuses on the **recent trajectory** of distance changes

**`distance_to_front_vehicle` across timesteps:**
- **t7, t6, t5, t4** (~0.01-0.013) - **MOST IMPORTANT**
- **t3, t2, t1** (~0.006-0.007) - **MODERATE**
- **t0** (~0.003) - **LEAST IMPORTANT**
- **Pattern**: More recent timesteps are more important
- **Interpretation**: 
  - Current and very recent front vehicle distance matters most
  - Older information is less relevant

**`traffic_density` across timesteps:**
- All timesteps show **low importance** (~0.003-0.005)
- **Pattern**: Relatively uniform across time
- **Interpretation**: Traffic density is a background factor, not a primary driver

**`class_id` across timesteps:**
- All timesteps show **very low importance** (~0.000)
- **Interpretation**: Vehicle type doesn't matter at any point in the sequence

### Business Insight:
- **Temporal patterns matter**: The model uses recent history (t1-t6) more than very recent (t7) or old (t0) information
- **Distance to stop line has a "sweet spot"**: Middle timesteps are most informative
- **Front vehicle distance is time-sensitive**: More recent information is more valuable
- **Speed and TTC remain unimportant**: Even when considering temporal patterns, these features don't contribute

---

## Summary of Key Insights

### ✅ What the Model is Doing Well:
1. **Spatial awareness**: Strongly uses distance to stop line and front vehicle
2. **Temporal reasoning**: Leverages recent history effectively (t1-t6)
3. **Consistent patterns**: Clear feature importance hierarchy

### ⚠️ Potential Issues:
1. **Speed is ignored**: Current speed should theoretically be important but isn't
2. **TTC is ignored**: Time-to-collision should be a safety factor but isn't
3. **Vehicle type doesn't matter**: May indicate insufficient data diversity or feature encoding issues

### 🔍 Recommendations:
1. **Investigate speed feature**: 
   - Check if speed values are properly normalized
   - Verify speed data quality in training set
   - Consider feature engineering (e.g., speed relative to limit)

2. **Investigate TTC feature**:
   - Check if TTC calculations are correct
   - Verify TTC data distribution (may be too sparse)
   - Consider alternative safety metrics

3. **Feature engineering opportunities**:
   - Create interaction features (speed × distance)
   - Add derived features (deceleration rate, stopping distance)
   - Consider relative features (distance / speed = time to stop line)

4. **Model improvement**:
   - The model may benefit from attention mechanisms to better use speed/TTC
   - Consider feature importance regularization to encourage use of all features

---

## How to Use These Insights

### For Model Improvement:
- Focus on fixing speed and TTC feature extraction/engineering
- Consider adding interaction terms
- Review data quality for underutilized features

### For Deployment:
- **Primary factors to monitor**: Distance to stop line, distance to front vehicle
- **Temporal window**: Recent 6 timesteps are most critical
- **Less important**: Vehicle type, current speed (in current model)

### For Safety:
- The model's reliance on distance over speed may be a concern
- Consider adding speed-based safety constraints in production
- Monitor for cases where high speed + close distance should trigger STOP but don't

