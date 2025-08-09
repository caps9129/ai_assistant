import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 您的 OpenAI API Key 應設定在環境變數中
client = OpenAI()

# 1. 指定一個您確定沒問題的本地音訊檔案路徑
audio_file_path = "/home/aiden/Desktop/aiden/spanish.mp3"  # <--- 請將檔名換成您自己的檔案

# 2. 檢查檔案是否存在
if not os.path.exists(audio_file_path):
    print(f"❌ 錯誤: 找不到測試檔案 '{audio_file_path}'")
    print("請確認檔案存在於專案目錄下，並且包含真實的語音。")
else:
    print(f"正在嘗試轉寫檔案: '{audio_file_path}'...")
    try:
        # 3. 使用 'with open' 來讀取檔案，並直接傳給 API
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,  # 直接傳遞檔案控制代碼
                response_format="text"
            )

        print("✅ 轉寫成功！")
        print(f"結果: {transcript}")

    except Exception as e:
        print(f"❌ OpenAI 轉寫失敗: {e}")
