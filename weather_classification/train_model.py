import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib

# ==========
# 參數設定
# ==========
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "ACDC_dataset"  # training dataset folder path

data_dir = pathlib.Path(DATA_DIR)

# ==========
# 建立資料集
# ==========
print("[INFO] Loading datasets...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)
print("Datasets loaded.\n")

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# 讓資料 pipeline 比較順
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==========
# 資料增強（Data Augmentation）
# ==========
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

# ==========
# 建立 MobileNetV2 + 自訂分類頭
# ==========
# 使用 ImageNet 預訓練權重（weight），不包含最上面的分類層
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=0.5,
)

# 先凍結 backbone，只訓練新頭（classification head）
base_model.trainable = False

# Keras Functional API 建模
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

# 1. 資料增強
x = data_augmentation(inputs)

# 2. MobileNetV2 預處理（跟訓練 ImageNet 時同一種 normalization）
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# 3. 特徵抽取 backbone
x = base_model(x, training=False)  # training=False 很重要，避免 BN（Batch Normalization）亂動統計量

# 4. GlobalAveragePooling 把 feature map 壓成一個向量
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

# 5. 自訂 Dense + Dropout（這整段就是「分類頭」的一部分）
x = layers.Dropout(0.2, name="dropout")(x)
outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="weather_mobilenetv2")

model.summary()

# ==========
# 編譯模型
# ==========
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",  # 因為 label 是整數類別 id
    metrics=["accuracy"],
)

# ==========
# Callbacks：存最佳模型＋早停
# ==========
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="weather_mobilenetv2_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
]

# ==========
# 訓練
# ==========
print(f"\n[INFO] Start training for {EPOCHS} epochs...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ==========
# 儲存整個模型（方便之後轉 TFLite / Realtek SDK）
# ==========
model.save("weather_mobilenetv2_best.h5", include_optimizer=False)
print("\n[INFO] Model saved as 'weather_mobilenetv2_best.h5'")