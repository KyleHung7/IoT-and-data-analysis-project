# 天氣辨識

透過行車影像擷取出的圖片，使用 keras 訓練模型，辨識天氣狀況為 好天氣（clear）、雨天（rain）、起霧（fog）、夜晚（night）、下雪（snow）。

模型檔案為 [weather_mobilenetv2_best.h5](weather_mobilenetv2_best.h5)。

另，使用 [氣象資料開放平台 API](https://opendata.cwa.gov.tw/devManual/datalist) L-003，可藉由經緯度查詢最接近的氣象觀測站的天氣狀況。

## API 取得天氣資訊

```bash
python3 weather_api.py
```

使用 `get_weather_condition_api(latitude, longitude, API_KEY)` 以取得觀測站回傳相關資訊，及天氣狀況 label。

使用過去一小時降雨量輔助判斷天氣狀況。

## 模型架構

使用經過 ImageNet 預訓練的 MobileNetV2 架構作為 backbone，套用自定義的分類頭，輸出 5 個不同的 label 對應的機率。

## 資料集

使用 [ACDC (Adverse Conditions Dataset with Correspondences)](https://acdc.vision.ee.ethz.ch) 資料集進行模型訓練。此資料集包含不同天氣情況下的行車記錄器照片。

若要使用此資料集訓練的模型，需引用以下論文：

Christos Sakaridis, Dengxin Dai, and Luc Van Gool. "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding". Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

## 資料處理

### 檔案架構處理

在模型訓練中，將 `"*_ref"` 的資料夾下的檔案統一整理至 `"ACDC_dataset/clear"`，其餘資料夾下的檔案統一整理至各自分類的資料夾下。

### 資料預處理

訓練時，對圖片做以下處理：

- 將輸入影像尺寸調整為 160 * 160 * 3（RGB）

- 訓練階段的資料增強（Data Augmentation）：

  - 隨機水平翻轉：`RandomFlip("horizontal")`

  - 小幅度隨機旋轉：`RandomRotation(0.05)`

  - 隨機縮放：`RandomZoom(0.1)`

  - 隨機對比度調整：`RandomContrast(0.1)`

推理時，對圖片做以下處理：

- 將輸入影像尺寸調整為 160 * 160 * 3（RGB）

## 使用方式

### 使用 virtual environment，並安裝依賴套件

```bash
python3 -m venv .venv
pip install -r requirements.txt
```

因應 AMB82 MINI 的使用，將 python 版本設為 3.11.14，tensorflow 版本設為 2.14.0。

### 模型訓練
```bash
python train_model.py
```

需將 `ACDC_dataset` 資料夾放在與 `train_model.py` 同一資料夾下。

訓練後會生成 model 檔案 `weather_mobilenetv2_best.keras`。

### 模型推理（demo）

```bash
python classification_demo.py
```

將要辨識的圖片放在 `images` 資料夾下，此資料夾需與 `demo_classify.py` 在同一資料夾下。僅會辨識 `"*.jpg", "*.jpeg", "*.png", "*.bmp"` 的檔案。

辨識結果會顯示在圖片左上角；按 q 可結束程式，其他任意鍵切換到下一張。