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

# 設定學術風格繪圖參數
sns.set(style="ticks", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 資料載入與前處理 ---
print("Initializing Advanced Analysis...")
file_log = 'speed_log.csv'
file_sensor = 'merged_sensor_data.csv'

if not os.path.exists(file_log) or not os.path.exists(file_sensor):
    print(f"❌ Critical Error: Input files not found.")
    exit()

df_cam = pd.read_csv(file_log)
df_ameba = pd.read_csv(file_sensor)

# 清理欄位
df_cam.columns = df_cam.columns.str.strip()
df_ameba.columns = df_ameba.columns.str.strip()

# ==================================================
# 【關鍵修改】在這裡設定你在圖表上想要顯示的專業名稱
# ==================================================
df_cam['Device'] = 'Intersection Camera'  # 原 speed_log
df_ameba['Device'] = 'Ameba 82 mini'      # 原 Sensor_Data
# ==================================================

df_all = pd.concat([df_cam, df_ameba], axis=0, ignore_index=True)

# 定義關鍵物理特徵 (除去座標 x, y，專注於動力學特徵)
numeric_features = ['speed_kmh', 'distance_to_stop_line', 'ttc', 'traffic_density']
categorical_features = ['vehicle_type', 'traffic_light_status']

# Pipeline 建置
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # 改用 mean 填補，減少 0 的偏差
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

# --- 2. 降維與分群 (PCA + K-means) ---
print("Executing PCA & Clustering algorithms...")
X_processed = preprocessor.fit_transform(df_all)

# PCA (保留解釋力)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# 取得 PCA components (特徵向量)，用於後續解釋
pca_components = pca.components_[:, :len(numeric_features)]

# K-means (設 k=3: 預期分為 靜止等待、減速接近、全速通過)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)

df_all['PCA_1'] = X_pca[:, 0]
df_all['PCA_2'] = X_pca[:, 1]
df_all['Cluster'] = clusters

# =========================================
#  圖表 1: 雙視角比較圖 (修正標籤版)
# =========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Comparative Analysis of Latent Space Distributions', fontsize=16, weight='bold')

# Subplot 1: Semantic Clustering
scatter1 = sns.scatterplot(
    data=df_all, x='PCA_1', y='PCA_2', hue='Cluster',
    palette='viridis', s=80, alpha=0.8, ax=axes[0], edgecolor='k'
)
axes[0].set_title('Behavioral Clustering (K-means)', weight='bold')
axes[0].set_xlabel('Principal Component 1 (Kinematics)')
axes[0].set_ylabel('Principal Component 2 (Context)')

# Subplot 2: Device Source
# 這裡 hue='Device' 會使用我們前面設定好的新名稱
# 使用高對比顏色：Intersection Camera (灰/黑) vs Ameba (亮色) 以突顯差異
custom_palette = {'Intersection Camera': '#636e72', 'Ameba 82 mini': '#0984e3'}
scatter2 = sns.scatterplot(
    data=df_all, x='PCA_1', y='PCA_2', hue='Device',
    palette=custom_palette, s=80, alpha=0.7, ax=axes[1], edgecolor='None'
)
axes[1].set_title('Device Coverage Comparison', weight='bold')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')

plt.tight_layout()
plt.savefig('advanced_comparison_plot.png', dpi=300, bbox_inches='tight')
print("✅ Generated: advanced_comparison_plot.png")


# =========================================
#  圖表 2: PCA 特徵向量圖 (Feature Loadings)
# =========================================
plt.figure(figsize=(8, 6))
plt.title('PCA Feature Loadings (Eigenvectors)', fontsize=14, weight='bold')
plt.axvline(0, color='grey', linestyle='--')
plt.axhline(0, color='grey', linestyle='--')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('PC1 Loading (Contribution)')
plt.ylabel('PC2 Loading (Contribution)')

# 畫箭頭
for i, feature in enumerate(numeric_features):
    plt.arrow(0, 0, pca_components[0, i], pca_components[1, i],
              color='r', alpha=0.8, head_width=0.03)
    plt.text(pca_components[0, i]*1.15, pca_components[1, i]*1.15,
             feature, color='darkred', ha='center', va='center', fontsize=11, weight='bold')

plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('pca_loadings_analysis.png', dpi=300)
print("✅ Generated: pca_loadings_analysis.png")


# =========================================
#  圖表 3: 專業總結表 (修正: 移除來源佔比)
# =========================================
# 計算統計量
summary = df_all.groupby('Cluster').agg({
    'speed_kmh': ['mean', 'std'],
    'distance_to_stop_line': ['mean', 'std'],
    'ttc': 'mean',
    'traffic_density': 'mean'
}).round(2)

# 扁平化欄位名稱
summary.columns = [
    'Speed Avg (km/h)', 'Speed Std',
    'Dist. Avg (m)', 'Dist. Std',
    'TTC Avg (s)', 'Density Avg'
]

# 加入 Count
summary['Count'] = df_all['Cluster'].value_counts().sort_index()

# 調整欄位順序
cols = ['Count', 'Speed Avg (km/h)', 'Speed Std', 'Dist. Avg (m)', 'TTC Avg (s)', 'Density Avg']
summary = summary[cols]
summary.index = [f'Cluster {i}' for i in summary.index]

# 繪圖
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('tight')
ax.axis('off')

# 表格內容
table_data = summary.values
col_labels = summary.columns
row_labels = summary.index

table = ax.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels,
                 cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# 專業配色與格式化
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('white')
    if row == 0:
        cell.set_facecolor('#2d3436') # 深鐵灰 Header
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.18)
    elif col == -1:
        cell.set_facecolor('#dfe6e9') # Index 淺灰
        cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor('#f5f6fa')
        # 重點強調：如果速度標準差很大，標示出來 (代表該群行為不穩定)
        if col == 2 and summary.values[row-1][2] > 10:
            cell.set_text_props(color='#d63031', weight='bold') # 警示紅

plt.title('Cluster Physical Characteristics Profile', fontsize=14, weight='bold', y=1.05)
plt.savefig('professional_summary_table.png', dpi=300, bbox_inches='tight')
print("✅ Generated: professional_summary_table.png")

print("\n分析完成。")