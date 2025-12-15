import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import os

# ==========================================
#  1. Style & Font Settings (Fixed)
# ==========================================
# Use a robust font list to avoid "font not found" errors
# Priority: Arial -> DejaVu Sans (default safe) -> Generic sans-serif
sns.set(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Data Loading ---
print("Initializing Analysis...")
file_log = 'speed_log.csv'
file_sensor = 'merged_sensor_data.csv'

if not os.path.exists(file_log) or not os.path.exists(file_sensor):
    print(f"❌ Critical Error: Input files not found.")
    exit()

df_cam = pd.read_csv(file_log)
df_ameba = pd.read_csv(file_sensor)

# Clean columns
df_cam.columns = df_cam.columns.str.strip()
df_ameba.columns = df_ameba.columns.str.strip()

# Set Device Names (English)
df_cam['Device'] = 'Intersection Camera'
df_ameba['Device'] = 'Ameba 82 mini'

df_all = pd.concat([df_cam, df_ameba], axis=0, ignore_index=True)

# Define Features
numeric_features = ['speed_kmh', 'distance_to_stop_line', 'ttc', 'traffic_density']
categorical_features = ['vehicle_type', 'traffic_light_status']

# Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- PCA & Clustering ---
print("Executing PCA & Clustering...")
X_processed = preprocessor.fit_transform(df_all)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
pca_components = pca.components_[:, :len(numeric_features)]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)

df_all['PCA_1'] = X_pca[:, 0]
df_all['PCA_2'] = X_pca[:, 1]
df_all['Cluster'] = clusters


# ==============================================================================
#  Prepare PCA Loadings Data
# ==============================================================================
try:
    feature_names_raw = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names_raw = numeric_features + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))

# Clean feature names for better display
clean_feature_names = [f.replace('num__', '').replace('cat__', '').replace('traffic_light_status', 'Light').replace('vehicle_type', 'Type') for f in feature_names_raw]

pca_loadings_df = pd.DataFrame(
    pca.components_.T, 
    index=clean_feature_names,
    columns=['PC1 (Kinematics)', 'PC2 (Context)']
)

# ==============================================================================
#  Chart 1: Professional Diverging Bar Chart (PCA Loadings) - ENGLISH
# ==============================================================================
print("Generating professional PCA loadings bar chart...")

def plot_loadings_bar(ax, pc_series, title, subtitle_pos, subtitle_neg):
    # Sort by absolute value
    sorted_series = pc_series.abs().sort_values(ascending=True)
    original_values = pc_series[sorted_series.index]
    
    # Colors: Blue for Positive, Red for Negative
    colors = ['#0984e3' if x > 0 else '#d63031' for x in original_values]
    
    # Draw bars
    bars = ax.barh(original_values.index, original_values.values, color=colors, alpha=0.8, edgecolor='none')
    
    # Add center line
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    
    # Add value labels
    for bar, value in zip(bars, original_values.values):
        width = bar.get_width()
        label_x_pos = width if width > 0 else width - 0.02
        ha_align = 'left' if width > 0 else 'right'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{value:+.3f}', 
                va='center', ha=ha_align, fontsize=10, fontweight='bold', color='#2d3436')

    # Titles and Annotations (ALL ENGLISH)
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.text(0.02, 1.02, subtitle_neg, transform=ax.transAxes, color='#d63031', fontsize=10, ha='left', fontweight='bold')
    ax.text(0.98, 1.02, subtitle_pos, transform=ax.transAxes, color='#0984e3', fontsize=10, ha='right', fontweight='bold')
    
    ax.set_xlabel('Feature Weight (Contribution)', fontsize=11, labelpad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=False, right=True, top=True)

# Create figure
fig_bars, axes_bars = plt.subplots(2, 1, figsize=(10, 14))

# Plot PC1
plot_loadings_bar(
    axes_bars[0], 
    pca_loadings_df['PC1 (Kinematics)'], 
    title='(a) PC1 Top Influencers (Kinematics)',
    subtitle_pos='Positive (+): Moves point RIGHT',
    subtitle_neg='Negative (-): Moves point LEFT'
)

# Plot PC2
plot_loadings_bar(
    axes_bars[1], 
    pca_loadings_df['PC2 (Context)'], 
    title='(b) PC2 Top Influencers (Context)',
    subtitle_pos='Positive (+): Moves point UP',
    subtitle_neg='Negative (-): Moves point DOWN'
)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig('pca_loadings_bars_english.png', dpi=300, bbox_inches='tight')
print("✅ Generated: pca_loadings_bars_english.png")


# ==============================================================================
#  Chart 2: Scatter Plots (Cluster & Device) - ENGLISH
# ==============================================================================
# Reset style for scatter plots
sns.set(style="ticks", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Comparative Analysis of Latent Space Distributions', fontsize=16, weight='bold')

# Scatter 1
sns.scatterplot(
    data=df_all, x='PCA_1', y='PCA_2', hue='Cluster',
    palette='viridis', s=80, alpha=0.8, ax=axes[0], edgecolor='k'
)
axes[0].set_title('Behavioral Clustering (K-means)', weight='bold')
axes[0].set_xlabel('Principal Component 1 (Kinematics)')
axes[0].set_ylabel('Principal Component 2 (Context)')
axes[0].legend(title='Cluster ID')

# Scatter 2
custom_palette = {'Intersection Camera': '#636e72', 'Ameba 82 mini': '#0984e3'}
sns.scatterplot(
    data=df_all, x='PCA_1', y='PCA_2', hue='Device',
    palette=custom_palette, s=80, alpha=0.7, ax=axes[1], edgecolor='None'
)
axes[1].set_title('Device Coverage Comparison', weight='bold')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].legend(title='Data Source')

plt.tight_layout()
plt.savefig('advanced_comparison_plot_english.png', dpi=300, bbox_inches='tight')
print("✅ Generated: advanced_comparison_plot_english.png")


# ==============================================================================
#  Chart 3: Summary Table - ENGLISH
# ==============================================================================
summary = df_all.groupby('Cluster').agg({
    'speed_kmh': ['mean', 'std'],
    'distance_to_stop_line': ['mean', 'std'],
    'ttc': 'mean',
    'traffic_density': 'mean'
}).round(2)

summary.columns = [
    'Speed Avg (km/h)', 'Speed Std',
    'Dist. Avg (m)', 'Dist. Std',
    'TTC Avg (s)', 'Density Avg'
]

summary['Count'] = df_all['Cluster'].value_counts().sort_index()
cols = ['Count', 'Speed Avg (km/h)', 'Speed Std', 'Dist. Avg (m)', 'TTC Avg (s)', 'Density Avg']
summary = summary[cols]
summary.index = [f'Cluster {i}' for i in summary.index]

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=summary.values, colLabels=summary.columns, rowLabels=summary.index,
                 cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('white')
    if row == 0:
        cell.set_facecolor('#2d3436')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.18)
    elif col == -1:
        cell.set_facecolor('#dfe6e9')
        cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor('#f5f6fa')
        if col == 2 and summary.values[row-1][2] > 10:
            cell.set_text_props(color='#d63031', weight='bold')

plt.title('Cluster Physical Characteristics Profile', fontsize=14, weight='bold', y=1.05)
plt.savefig('professional_summary_table_english.png', dpi=300, bbox_inches='tight')
print("✅ Generated: professional_summary_table_english.png")

print("\nAnalysis Completed Successfully.")