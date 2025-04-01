import cv2
import ollama
import re
import os
import json
import sys
import wikipedia
from difflib import get_close_matches
from wit import Wit
import pyaudio
import wave
import io

# -----------------------------
# 參數設定與全域變數
# -----------------------------
# 讀取檔案中的 Token
token_file = open("wit_token.txt", "r")
WIT_ACCESS_TOKEN = token_file.read().strip()
token_file.close()

# 使用 Token 初始化 Wit
client = Wit(WIT_ACCESS_TOKEN)

FRUIT_JSON_PATH = "/opt/NanoLLM/ollama_host/fruit_dataset.json"

# 允許辨識的水果清單
ALLOWED_FRUITS = [
    "Apple", "Banana", "Grape", "Kiwi", "Mango", "Orange",
    "Strawberry", "Chickoo", "Cherry", "Watermelon",
    "Guava", "Pineapple", "Cantaloupe"
]

# -----------------------------
# 使用 PyAudio 錄音
# -----------------------------
def record_audio_pyaudio(duration=3, filename="voice_command.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("開始錄音...")
    frames = []
    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("錄音結束")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

# -----------------------------
# Wit.ai 語音辨識
# -----------------------------
def recognize_speech_with_wit(audio_file, access_token=WIT_ACCESS_TOKEN):
    client = Wit(access_token)
    with open(audio_file, 'rb') as f:
        response = client.speech(f, {'Content-Type': 'audio/wav'})
    return response.get('text', None)

# -----------------------------
# 幫 Wikipedia 回傳的內容，產出僅有兩行的精簡描述
# 第一行: nutrition: ...
# 第二行: health: ...
# -----------------------------
def shorten_wiki_text(main_text):
    """
    1. 移除 [1], [2] 這類參考符號 & 多餘問號或空行
    2. 透過 ollama 請求僅輸出兩行:
       nutrition: <一句關於該水果的營養資訊>
       health: <一句關於該水果的健康益處>
    """
    # 清理雜訊
    text_no_refs = re.sub(r"\[\d+\]", "", main_text)
    text_no_refs = re.sub(r"[\?]{2,}", "", text_no_refs)
    text_no_refs = re.sub(r"\n+", " ", text_no_refs)

    prompt = f"""請閱讀以下水果資訊，並只用兩行輸出：
nutrition: <水果的營養相關描述>
health: <水果的健康益處描述>
請確保只輸出上述兩行，不要添加任何其他多餘文字或前綴。
以下是原始內容：
{text_no_refs}
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    two_lines = response["message"]["content"].strip()
    
    # 新增：容錯機制（用正規表示法找兩行）
    nutrition_match = re.search(r"nutrition:\s*(.+)", two_lines, re.IGNORECASE)
    health_match = re.search(r"health:\s*(.+)", two_lines, re.IGNORECASE)

    nutrition_line = f"nutrition: {nutrition_match.group(1).strip()}" if nutrition_match else "nutrition: 無"
    health_line = f"health: {health_match.group(1).strip()}" if health_match else "health: 無"

    return nutrition_line, health_line

# -----------------------------
# 辨識水果 (OpenCV frame or image path)
# -----------------------------
def identify_fruit(frame=None, image_path=None):
    if frame is not None:
        temp_path = "current_frame.jpg"
        cv2.imwrite(temp_path, frame)
        image_source = temp_path
    elif image_path is not None:
        if not os.path.exists(image_path):
            print(f"❌ 找不到圖片 {image_path}。")
            return None
        image_source = image_path
    else:
        print("❌ 未提供圖片來源。")
        return None

    llava_prompt = """
    Please analyze this image and output only a single fruit name (for example,
    "Apple", "Banana", "Grape", "Kiwi", "Mango", "Orange", "Strawberry",
    "Chickoo", "Cherry", "Watermelon", "Guava", "Pineapple", "Cantaloupe").
    Only respond with the fruit name without any extra characters, punctuation,
    numbers, or explanation.
    """

    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": llava_prompt,
            "images": [image_source]
        }]
    )
    fruit_result = response["message"]["content"].strip()
    match = re.search(r"\*\*Answer:\*\*\s*(\w+)", fruit_result)
    recognized = match.group(1) if match else fruit_result
    recognized = recognized.title()
    recognized = re.sub(r"[^A-Za-z ]", "", recognized).strip()

    if recognized not in ALLOWED_FRUITS:
        print(f"辨識結果 '{recognized}' 不在允許清單中。")
        return None

    return recognized

# -----------------------------
# 從 Wikipedia 獲取水果資訊 (自動拆分成 nutrition 與 health 兩行)
# -----------------------------
def fetch_fruit_info_online(fruit_name):
    try:
        wikipedia.set_lang("en")
        query_name = fruit_name if fruit_name.lower() != "pear" else "Pear (fruit)"

        # 只抓 2 句 summary
        main_summary = wikipedia.summary(query_name, sentences=2)

        # 從完整頁面擷取 nutrition 區塊（若有）
        page = wikipedia.page(query_name)
        content = page.content
        idx = content.find("Nutrition")
        nutrition_excerpt = ""
        if idx != -1:
            nutrition_excerpt = content[idx:idx+300]

        combined_text = main_summary + "\n" + nutrition_excerpt

        # 讓模型只輸出兩行
        nutrition_line, health_line = shorten_wiki_text(combined_text)

        # 回傳兩行分別給 dictionary
        return {
            "nutrition": nutrition_line,
            "health_benefits": health_line
        }

    except wikipedia.DisambiguationError as e:
        print(f"⚠️ Multiple results: {e.options}")
        return None
    except Exception as e:
        print(f"⚠️ Wikipedia 擷取失敗: {e}")
        return None

# -----------------------------
# 先查 JSON，若無，再查 Wikipedia
# -----------------------------
def get_fruit_info(fruit_name):
    if not os.path.exists(FRUIT_JSON_PATH):
        print(f"❌ 找不到 {FRUIT_JSON_PATH}，請確認路徑。")
        sys.exit(1)

    with open(FRUIT_JSON_PATH, "r", encoding="utf-8") as file:
        fruit_data = json.load(file)

    info = next((f for f in fruit_data if f["fruit"].lower() == fruit_name.lower()), None)
    if info:
        return info

    print(f"⚠️ 資料庫中無 '{fruit_name}' 的資訊，改從 Wikipedia 搜尋...")
    wiki_info = fetch_fruit_info_online(fruit_name)
    if wiki_info:
        return {
            "fruit": fruit_name,
            "nutrition": wiki_info.get("nutrition", "nutrition: 無"),
            "health_benefits": wiki_info.get("health_benefits", "health: 無"),
        }
    return None

# -----------------------------
# 使用 LLM 針對水果作 Q&A
# -----------------------------
def query_ai_for_fruit(fruit_name, fruit_info, query_type="general", question=None):
    """
    - query_type 可為 'calories', 'vitamins', 'health_benefits', 或 'general'
    - 若是 general，則將所有資訊帶入 prompt，讓模型自由回答
    """
    # 以下僅示範，可依實際需求修改解析邏輯
    if query_type == "calories":
        return "解析卡路里資訊 (示範)"
    elif query_type == "vitamins":
        return "解析維生素資訊 (示範)"
    elif query_type == "health_benefits":
        return "解析健康益處 (示範)"
    else:
        prompt = f"""You are an expert in fruits, including their nutritional value and health benefits.
Please answer the user's question based on the following information. If the provided data is insufficient,
you may incorporate your general knowledge about this fruit. Please respond concisely in English.

Fruit name: {fruit_name}
Nutrition info: {fruit_info.get('nutrition', 'nutrition: Not available')}
Health benefits: {fruit_info.get('health_benefits', 'health: Not available')}

The user's question is: "{question}"
"""
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

# -----------------------------
# 顯示水果資訊 (兩行)
# -----------------------------
def display_fruit_info(fruit_info):
    if not fruit_info:
        print("⚠️ 無水果資訊。")
        return
    print(f"\n🍎 Fruit Info:")
    print(f"Name: {fruit_info.get('fruit', 'Unknown')}")
    print(f"{fruit_info.get('nutrition', 'nutrition: Not available')}")
    print(f"{fruit_info.get('health_benefits', 'health: Not available')}")

# -----------------------------
# OpenCV 文字換行輔助
# -----------------------------
def wrap_text(text, font, font_scale, thickness, max_width):
    lines = []
    words = text.split(' ')
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        (width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

# -----------------------------
# Voice Chat mode
# -----------------------------
def voice_chat(access_token, fruit_name_on_screen, local_fruit_info):
    print("Recording voice for 3 seconds...")
    audio_file = record_audio_pyaudio(duration=3)
    question = recognize_speech_with_wit(audio_file, access_token)
    if question:
        print("Wit.ai recognized question:", question)
        if "calorie" in question.lower() or "卡路里" in question:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="calories")
        elif "vitamin" in question.lower() or "維生素" in question:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="vitamins")
        elif "health" in question.lower() or "益處" in question:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="health_benefits")
        else:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, question=question)
        print("AI answer:", answer)
    else:
        print("No speech detected.")

# -----------------------------
# 結合影像辨識 + 語音詢問
# -----------------------------
def combined_operation_with_frame(frame, access_token):
    fruit_name_on_screen = identify_fruit(frame=frame)
    if not fruit_name_on_screen:
        print("辨識失敗。")
        return
    print("辨識結果：", fruit_name_on_screen)

    local_fruit_info = get_fruit_info(fruit_name_on_screen)
    if local_fruit_info:
        display_fruit_info(local_fruit_info)
    else:
        print("無法取得該水果相關資訊。")

    audio_file = record_audio_pyaudio(duration=3)
    voice_command = recognize_speech_with_wit(audio_file, access_token)
    if voice_command:
        print("Wit.ai 識別的語音：", voice_command)
    else:
        print("未偵測到語音。")
        return

    # 按關鍵字判斷
    if "calorie" in voice_command.lower() or "卡路里" in voice_command:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="calories")
    elif "vitamin" in voice_command.lower() or "維生素" in voice_command:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="vitamins")
    elif "health" in voice_command.lower() or "益處" in voice_command:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="health_benefits")
    else:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, question=voice_command)

    print("AI answer:", answer)

# -----------------------------
# 啟動 Webcam 模式
# -----------------------------
def run_webcam_mode():
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)  # 視需求調整攝影機編號
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("無法開啟攝影機。")
        sys.exit(1)

    fruit_name_on_screen = ""
    nutrition_on_screen = ""
    health_benefits_on_screen = ""
    local_fruit_info = {}
    voice_command = ""

    print("Press 'o' to identify the fruit, 's' for voice recognition,")
    print("Press 'c' for voice chat, 'x' for combined operation, 'q' to quit.")

    cv2.namedWindow("Fruit Information", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fruit Information", 850, 600)
    cv2.moveWindow("Fruit Information", 100, 200)

    max_width = 800
    fruit_name_y_pos = 50
    y_pos = 100

    access_token = WIT_ACCESS_TOKEN

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 顯示 Fruit 名稱
        fruit_lines = wrap_text(f"Fruit: {fruit_name_on_screen}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2, max_width)
        line_y = fruit_name_y_pos
        for line in fruit_lines:
            cv2.putText(frame, line, (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            line_y += 25

        # 顯示 nutrition
        nutrition_lines = wrap_text(nutrition_on_screen, cv2.FONT_HERSHEY_SIMPLEX, 1, 1, max_width)
        ny = y_pos
        for line in nutrition_lines:
            cv2.putText(frame, line, (10, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
            ny += 25

        # 顯示 health
        ny += 30
        health_lines = wrap_text(health_benefits_on_screen, cv2.FONT_HERSHEY_SIMPLEX, 1, 1, max_width)
        for line in health_lines:
            cv2.putText(frame, line, (10, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
            ny += 25

        # 顯示語音內容
        cv2.putText(frame, f"Voice: {voice_command}", (10, ny + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

        cv2.imshow("Fruit Information", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            fruit_name_on_screen = identify_fruit(frame=frame)
            if fruit_name_on_screen:
                local_fruit_info = get_fruit_info(fruit_name_on_screen)
                if local_fruit_info:
                    nutrition_on_screen = local_fruit_info.get("nutrition", "nutrition: 無")
                    health_benefits_on_screen = local_fruit_info.get("health_benefits", "health: 無")
                else:
                    nutrition_on_screen = "nutrition: 無"
                    health_benefits_on_screen = "health: 無"
            # 重置顯示位置
            ny = y_pos
        elif key == ord('s'):
            audio_file = record_audio_pyaudio(duration=3)
            recognized = recognize_speech_with_wit(audio_file, access_token)
            if recognized:
                voice_command = recognized
                print("Wit.ai recognized voice:", voice_command)
            else:
                voice_command = "No voice command detected."
            ny = y_pos
        elif key == ord('c'):
            print(f"\nVoice Chat Mode about {fruit_name_on_screen}:")
            voice_chat(access_token, fruit_name_on_screen, local_fruit_info)
            ny = y_pos
        elif key == ord('x'):
            combined_operation_with_frame(frame, access_token)
            ny = y_pos

    cap.release()
    cv2.destroyAllWindows()

def main():
    # 只啟動 Webcam 模式
    run_webcam_mode()

if __name__ == "__main__":
    main()
