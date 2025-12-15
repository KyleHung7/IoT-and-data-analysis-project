import os
import glob

import cv2
import numpy as np
import tensorflow as tf

# ========================
# 使用者需要修改的設定區
# ========================

# 你的模型路徑 (Keras SavedModel 或 .h5 皆可)
MODEL_PATH = "models/weather_mobilenetv2_best.h5"

# 影像輸入尺寸 (請和訓練模型時的一樣)
IMG_WIDTH = 160
IMG_HEIGHT = 160

# 天氣類別名稱 (順序要與模型輸出的 softmax 順序一致)
LABELS = [
    "clear",       # 晴天
    "night",       # 夜晚
    "not_clear",   # 雨天
]

# 測試圖片所在資料夾
IMAGE_DIR = "images"

# 是否顯示圖片視窗
SHOW_IMAGE = True

# 顯示視窗名稱
WINDOW_NAME = "Weather Classification Demo (From Files)"


# ========================
# 模型與前處理
# ========================

def load_weather_model(model_path: str):
    print(f"[INFO] 載入模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] 模型載入完成")
    print("      Input shape :", model.input_shape)
    print("      Output shape:", model.output_shape)
    return model


def preprocess_image(bgr_img: np.ndarray) -> np.ndarray:
    """
    將 OpenCV BGR 影像轉換為模型可接受的輸入格式。

    步驟：
    1. 調整大小至 (IMG_WIDTH, IMG_HEIGHT)
    2. BGR -> RGB (若訓練時使用 RGB)
    3. 轉為 float32
    4. 增加 batch 維度，變成 shape: (1, H, W, C)

    Parameters
    ----------
    bgr_img : np.ndarray
        從 OpenCV 取得的 BGR 影像

    Returns
    -------
    np.ndarray
        可丟進 Keras 模型的輸入張量 (batch size = 1)
    """
    # 調整大小
    img = cv2.resize(bgr_img, (IMG_WIDTH, IMG_HEIGHT))

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 轉 float32
    img = img.astype("float32")

    # 增加 batch 維度
    img = np.expand_dims(img, axis=0)
    print(f"  -> 輸入張量 shape: {img.shape}")
    print(img[0, 0:2, 0:2, :])  # 印出左上角 2x2 像素的數值供參考
    return img


def predict_weather(model: tf.keras.Model, preprocessed_img: np.ndarray):
    """
    使用模型進行單張影像的天氣分類。

    Parameters
    ----------
    model : tf.keras.Model
        已載入的 Keras 模型
    preprocessed_batch : np.ndarray
        經過 preprocess_image 處理後的輸入，shape = (1, H, W, C)

    Returns
    -------
    label : str
        預測類別名稱
    prob : float
        該類別的機率 (0 ~ 1)
    probs : np.ndarray
        所有類別的機率向量
    """
    probs = model.predict(preprocessed_img, verbose=0)[0]  # shape: (num_classes,)
    idx = int(np.argmax(probs))
    label = LABELS[idx] if idx < len(LABELS) else f"class_{idx}"
    prob = float(probs[idx])
    return label, prob, probs


# ========================
# 讀取檔案並分類
# ========================

def get_image_file_list(image_dir: str):
    """
    取得資料夾底下所有常見圖片副檔名的檔案清單。

    Parameters
    ----------
    image_dir : str
        圖片資料夾路徑

    Returns
    -------
    list[str]
        圖片檔案完整路徑列表
    """
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    file_list = []
    for p in patterns:
        file_list.extend(glob.glob(os.path.join(image_dir, p)))
    file_list.sort()
    return file_list


def main():
    # 檢查圖片資料夾是否存在
    if not os.path.isdir(IMAGE_DIR):
        print(f"[ERROR] 找不到圖片資料夾: {IMAGE_DIR}")
        print("        請建立此資料夾，並放入要測試的圖片檔。")
        return

    # 取得圖片清單
    image_files = get_image_file_list(IMAGE_DIR)
    if not image_files:
        print(f"[WARN] 在資料夾 {IMAGE_DIR} 中找不到任何圖片檔（jpg/jpeg/png/bmp）")
        return

    print(f"[INFO] 在 {IMAGE_DIR} 中找到 {len(image_files)} 張圖片")

    # 載入模型
    model = load_weather_model(MODEL_PATH)

    if SHOW_IMAGE:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # 逐張圖片做分類
    for img_path in image_files:
        print(f"\n[INFO] 處理圖片: {img_path}")

        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            print("[WARN] 無法讀取此圖片，略過")
            continue

        batch = preprocess_image(bgr_img)
        label, prob, probs = predict_weather(model, batch)

        # 在終端機印出結果
        print(f"  -> 預測結果: {label} (prob = {prob:.4f})")
        print(f"  -> 機率向量: {probs}")

        # 視需要顯示圖片視窗
        if SHOW_IMAGE:
            display_img = bgr_img.copy()
            text = f"{label} ({prob:.2f})"
            cv2.putText(
                display_img,
                text,
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 0, 255),  # 紅色文字 (BGR)
                4,
                cv2.LINE_AA
            )

            cv2.imshow(WINDOW_NAME, display_img)
            print("  -> 按任意鍵看下一張，或按 'q' 結束")

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                print("[INFO] 使用者要求提前結束")
                break

    if SHOW_IMAGE:
        cv2.destroyAllWindows()

    print("\n[INFO] 所有圖片處理完畢，程式結束")


if __name__ == "__main__":
    main()