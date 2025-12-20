import pyttsx3

# 初始化 TTS 引擎
engine = pyttsx3.init()

# 設定要轉換的文字
text = "生成語音中,請稍等。"

# 設定語速（可選）
engine.setProperty('rate', 140)

# 設定語音（選擇語音）
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # 可自行調整索引選不同聲音

# 將語音存成 WAV 檔
engine.save_to_file(text, 'output.wav')

# 執行轉換
engine.runAndWait()

print("語音已生成並存成 output.wav")
