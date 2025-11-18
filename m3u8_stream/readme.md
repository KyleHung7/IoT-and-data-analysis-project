# opencv with m3u8 stream

** 測試資料：https://tw.live/cam/?id=BOT243 **

## only_cv2.py

只使用OpenCV連接m3u8網址，獲取單偵資訊於`frame`

## use_ffmpeg.py

在`only_cv2.py`發現執行效率低下，因此改使用GPU硬體解碼，在筆電`i9-12900H`、`RTX 3050 Ti`測試發現`CPU`硬解效率大於`GPU`硬解