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
# 請將下面的 YOUR_WIT_ACCESS_TOKEN 替換為你在 Wit.ai 的存取權杖
WIT_ACCESS_TOKEN = ""
client = Wit(WIT_ACCESS_TOKEN)

# 統一水果資料庫的 JSON 檔案路徑（請確保此檔案存在或自行建立）
FRUIT_JSON_PATH = "/opt/NanoLLM/ollama_host/fruit_dataset.json"

# 記錄使用者問過的問題，防止重複回答
question_history = {}

# 全域變數，方便在 CLI 模式下更新水果資訊
fruit_name = ""
fruit_info = {}

# 允許辨識的水果清單
ALLOWED_FRUITS = ["Apple", "Banana", "Grape", "Kiwi", "Mango", "Orange", "Strawberry", "Chickoo", "Cherry"]

# -----------------------------
# 翻譯函式：將文字翻譯成繁體中文
# -----------------------------
def translate_to_zh(text):
    """
    使用 ollama 模型將輸入的文字翻譯成繁體中文
    """
    prompt = f"請將以下文字翻譯成繁體中文：\n\n{text}"
    response = ollama.chat(
        model="llama3",  # 或改用其他你認為適合的模型
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# -----------------------------
# PyAudio 錄音函式 (僅使用 PyAudio)
# -----------------------------
def record_audio_pyaudio(duration=3, filename="voice_command.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Wit.ai 推薦 16kHz
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("開始錄音...")
    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
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
# Wit.ai 語音辨識函式
# -----------------------------
def recognize_speech_with_wit(audio_file, access_token=WIT_ACCESS_TOKEN):
    client = Wit(access_token)
    with open(audio_file, 'rb') as f:
        response = client.speech(f, {'Content-Type': 'audio/wav'})
    return response.get('text', None)

# -----------------------------
# 水果辨識與資訊查詢函式
# -----------------------------
def identify_fruit(frame=None, image_path=None, confirm=True):
    """
    辨識水果名稱，輸入可以是攝影機捕捉的 frame 或圖片路徑，
    若 confirm 為 True 則請使用者確認辨識結果（CLI 模式）。
    僅接受 ALLOWED_FRUITS 清單中的水果，若辨識結果不在清單中，則請使用者手動輸入。
    """
    if frame is not None:
        temp_path = "current_frame.jpg"
        cv2.imwrite(temp_path, frame)
        image_source = temp_path
    elif image_path is not None:
        if not os.path.exists(image_path):
            print(f"❌ 找不到圖片 {image_path}，請檢查路徑。")
            return None
        image_source = image_path
    else:
        print("❌ 未提供圖片來源。")
        return None

    llava_prompt = """
    Please analyze this image and output only a single fruit name (for example, "Apple", "Banana", "Grape", "Kiwi", "Mango", "Orange", "Strawberry", "Chickoo", "Cherry").
    Only respond with the fruit name without any extra characters, punctuation, numbers, or explanation. If unsure, try to guess a similar fruit name.
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
    if match:
        recognized = match.group(1)
    else:
        recognized = fruit_result

    recognized = recognized.title()
    recognized = re.sub(r"[^A-Za-z ]", "", recognized).strip()

    # 檢查是否在允許清單中
    if recognized not in ALLOWED_FRUITS:
        print(f"辨識結果 '{recognized}' 不在允許清單中。")
        recognized = input(f"請從 {ALLOWED_FRUITS} 中輸入正確的水果名稱：").strip().title()
        if recognized not in ALLOWED_FRUITS:
            print("輸入錯誤，請確認後再試。")
            return None

    if confirm:
        user_confirm = input(f"🔍 模型辨識到：{recognized}，是否正確？ (yes/no): ").strip().lower()
        if user_confirm != "yes":
            recognized = input(f"請從 {ALLOWED_FRUITS} 中輸入正確的水果名稱：").strip().title()
            if recognized not in ALLOWED_FRUITS:
                print("輸入錯誤，請確認後再試。")
                return None
    return recognized

def fetch_fruit_info_online(fruit_name):
    """
    利用 wikipedia 從線上取得水果資訊，優先提供營養相關內容。
    """
    try:
        wikipedia.set_lang("en")
        query_name = fruit_name if fruit_name.lower() != "pear" else "Pear (fruit)"
        summary = wikipedia.summary(query_name, sentences=5)
        if "Nutrition" not in summary:
            page = wikipedia.page(query_name)
            content = page.content
            idx = content.find("Nutrition")
            if idx != -1:
                nutrition_excerpt = content[idx:idx+500]
                summary += "\n\nNutrition Info:\n" + nutrition_excerpt
        return summary
    except wikipedia.DisambiguationError as e:
        print(f"⚠️ 有多個結果：{e.options}")
        return None
    except Exception as e:
        print(f"⚠️ 從 Wikipedia 擷取資訊發生錯誤：{e}")
        return None

def get_fruit_info(fruit_name):
    """
    從 JSON 資料庫中取得水果資訊，找不到則嘗試線上查詢。
    """
    if not os.path.exists(FRUIT_JSON_PATH):
        print(f"❌ 找不到 {FRUIT_JSON_PATH}，請檢查路徑。")
        sys.exit(1)

    with open(FRUIT_JSON_PATH, "r", encoding="utf-8") as file:
        fruit_data = json.load(file)

    info = next((fruit for fruit in fruit_data if fruit_name.lower() == fruit["fruit"].lower()), None)
    if info:
        return info

    fruit_names = [fruit["fruit"] for fruit in fruit_data]
    matches = get_close_matches(fruit_name, fruit_names, n=1, cutoff=0.6)
    if matches:
        user_confirm = input(f"資料庫中找不到 '{fruit_name}'。是否是 '{matches[0]}'？ (yes/no): ").strip().lower()
        if user_confirm == "yes":
            return next((fruit for fruit in fruit_data if fruit["fruit"].lower() == matches[0].lower()), None)

    print(f"⚠️ 資料庫中無 '{fruit_name}' 的資訊，改從 Wikipedia 搜尋...")
    wiki_summary = fetch_fruit_info_online(fruit_name)
    if wiki_summary:
        user_confirm = input("是否採用此資訊？ (yes/no): ").strip().lower()
        if user_confirm == "yes":
            return {
                "fruit": fruit_name,
                "nutrition": wiki_summary,
                "health_benefits": wiki_summary,
            }
    return None

# -----------------------------
# 升級後的 query_ai_for_fruit 函式：使用較強模型回答自由問題
# -----------------------------
def query_ai_for_fruit(fruit_name, fruit_info, query_type="general", question=None):
    """
    根據使用者的詢問類型，從 fruit_info 中解析回應：
      - calories：卡路里資訊
      - vitamins：維生素資訊
      - health_benefits：健康益處
      - general：使用較強模型（例如 llama）回答自由問題
    """
    structured = "Per 100g:" in fruit_info.get('nutrition', "")
    if query_type == "calories":
        if structured:
            try:
                calories = fruit_info['nutrition'].split(':')[1].split(',')[0].strip()
                return f"{fruit_name} 每 100g 大約含有 {calories} 卡路里。"
            except Exception:
                return "⚠️ 解析卡路里資訊失敗。"
        else:
            match = re.search(r'(\d+)\s*kilocalories', fruit_info.get('nutrition', ""), re.IGNORECASE)
            if match:
                cal = match.group(1)
                return f"{fruit_name} 每 100g 約有 {cal} kilocalories。"
            else:
                return f"找不到 {fruit_name} 的卡路里資訊。"
    elif query_type == "vitamins":
        if structured:
            try:
                match = re.search(r"rich in (.+)", fruit_info.get('nutrition', ""), re.IGNORECASE)
                if match:
                    vitamins = match.group(1).strip()
                    return f"{fruit_name} 富含 {vitamins}。"
                else:
                    return f"找不到 {fruit_name} 的維生素資訊。"
            except Exception:
                return "⚠️ 解析維生素資訊失敗。"
        else:
            matches = re.findall(r'vitamin\s*([A-Za-z]+)', fruit_info.get('nutrition', ""), re.IGNORECASE)
            if matches:
                vitamins = ", ".join(sorted(set(matches)))
                return f"{fruit_name} 富含以下維生素：{vitamins}。"
            else:
                return f"找不到 {fruit_name} 的維生素資訊。"
    elif query_type == "health_benefits":
        if structured:
            return f"{fruit_name} 的健康益處：{fruit_info.get('health_benefits', '無健康益處資訊。')}"
        else:
            idx = fruit_info.get('nutrition', "").find("Research")
            if idx != -1:
                health_text = fruit_info.get('nutrition', "")[idx:]
                return f"{fruit_name} 的健康益處：{health_text}"
            else:
                return f"{fruit_name} 的健康益處：{fruit_info.get('nutrition', '')}"
    else:
        # 使用較強大的模型來回答自由問題
        prompt = f"""你是一位水果營養與健康專家，同時具備豐富的水果相關知識，
請根據以下資訊回答使用者的問題。如果資料庫中的資訊不足以回答，請根據你廣泛的水果知識補充回答，
並且請務必以繁體中文回覆，回答內容必須與水果主題密切相關且實用。
水果名稱：{fruit_name}
營養資料：{fruit_info.get('nutrition', '無')}
健康益處：{fruit_info.get('health_benefits', '無')}

問題是：「{question}」
"""
        response = ollama.chat(
            model="llama3",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response["message"]["content"]

# -----------------------------
# 輸出水果資訊函式
# -----------------------------
def display_fruit_info(fruit_info):
    """在 CLI 模式下印出水果資訊"""
    if not fruit_info:
        print("⚠️ 無水果資訊。")
        return
    print(f"\n🍎 水果資訊：")
    print(f"🔹 名稱: {fruit_info.get('fruit', 'Unknown')}")
    print(f"🔹 營養: {fruit_info.get('nutrition', 'N/A')}")
    print(f"🔹 健康益處: {fruit_info.get('health_benefits', 'N/A')}")

# -----------------------------
# 切換圖片函式
# -----------------------------
def change_image(new_image_path):
    """
    在 CLI 模式中切換圖片，更新 fruit_name 與 fruit_info，同時重置問答紀錄。
    """
    global fruit_name, fruit_info
    if os.path.exists(new_image_path):
        fruit_name = identify_fruit(image_path=new_image_path, confirm=True)
        fruit_info = get_fruit_info(fruit_name)
        display_fruit_info(fruit_info)
        question_history[fruit_name] = set()
        print("\n✅ 已切換水果，你現在可以詢問新水果相關問題！")
        return True
    else:
        print(f"❌ 找不到圖片 {new_image_path}，請檢查路徑。")
        return False

# -----------------------------
# 文字換行函式（用於 OpenCV 顯示）
# -----------------------------
def wrap_text(text, font, font_scale, thickness, max_width):
    """
    根據最大寬度將文字換行（用於 OpenCV 顯示）
    """
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
# voice_chat 函式：語音對話模式
# -----------------------------
def voice_chat(access_token, fruit_name_on_screen, local_fruit_info):
    """
    語音對話模式：錄製使用者語音問題，呼叫 Wit.ai 辨識後根據問題內容查詢水果資訊並印出回答。
    """
    print("請講出你的問題 (約 3 秒)...")
    audio_file = record_audio_pyaudio(duration=3)
    question = recognize_speech_with_wit(audio_file, access_token)
    if question:
        print("Wit.ai 辨識的問題：", question)
        if "calorie" in question.lower() or "卡路里" in question:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="calories")
        elif "vitamin" in question.lower() or "維生素" in question:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="vitamins")
        elif "health" in question.lower() or "益處" in question:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="health_benefits")
        else:
            answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, question=question)
        # 翻譯成繁體中文
        translated_answer = translate_to_zh(answer)
        print("🤖 AI 回答：", translated_answer)
    else:
        print("未偵測到語音問題。")

# -----------------------------
# 綜合操作函式：利用當前影像執行水果辨識、錄音識別與資訊查詢
# -----------------------------
def combined_operation_with_frame(frame, access_token):
    """
    綜合操作：利用傳入的 frame 執行水果辨識、錄音識別與資訊查詢，
    不需要重新開啟攝影機。
    """
    fruit_name_on_screen = identify_fruit(frame=frame, confirm=False)
    if not fruit_name_on_screen:
        print("水果辨識失敗。")
        return
    print("水果辨識結果：", fruit_name_on_screen)
    
    local_fruit_info = get_fruit_info(fruit_name_on_screen)
    if local_fruit_info:
        display_fruit_info(local_fruit_info)
    else:
        print("無法取得該水果相關資訊。")
    
    audio_file = record_audio_pyaudio(duration=3)
    voice_command = recognize_speech_with_wit(audio_file, access_token)
    if voice_command:
        print("Wit.ai 識別的語音內容：", voice_command)
    else:
        print("未偵測到語音內容。")
        return
    
    if "calorie" in voice_command.lower() or "卡路里" in voice_command:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="calories")
    elif "vitamin" in voice_command.lower() or "維生素" in voice_command:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="vitamins")
    elif "health" in voice_command.lower() or "益處" in voice_command:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="health_benefits")
    else:
        answer = query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, question=voice_command)
    translated_answer = translate_to_zh(answer)
    print("🤖 AI 回答：", translated_answer)

# -----------------------------
# 模式函式
# -----------------------------
def run_webcam_mode():
    """
    網路攝影機模式：使用 OpenCV 擷取即時影像，
    按下 'o' 辨識水果、's' 進行語音識別（僅更新畫面文字），
    'c' 進入語音對話模式，'x' 執行綜合操作（利用當前影像），
    'q' 離開程式。
    """
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
    # 設定解析度為 640x480 與偵率 15fps
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
    voice_command = ""  # 用於顯示語音識別結果

    print("按 'o' 進行水果辨識，按 's' 進行語音識別（僅更新畫面文字），")
    print("按 'c' 進入語音對話模式，按 'x' 執行綜合操作，按 'q' 離開。")

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

        # 顯示水果名稱、營養、健康益處及語音識別結果
        fruit_lines = wrap_text(f"Fruit: {fruit_name_on_screen}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2, max_width)
        for line in fruit_lines:
            cv2.putText(frame, line, (10, fruit_name_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

        nutrition_y_pos = y_pos
        nutrition_lines = wrap_text(nutrition_on_screen, cv2.FONT_HERSHEY_SIMPLEX, 1, 1, max_width)
        for line in nutrition_lines:
            cv2.putText(frame, line, (10, nutrition_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
            nutrition_y_pos += 25

        health_benefits_y_pos = nutrition_y_pos + 30
        health_benefits_lines = wrap_text(health_benefits_on_screen, cv2.FONT_HERSHEY_SIMPLEX, 1, 1, max_width)
        for line in health_benefits_lines:
            cv2.putText(frame, line, (10, health_benefits_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
            health_benefits_y_pos += 25

        cv2.putText(frame, f"Voice: {voice_command}", (10, health_benefits_y_pos + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

        cv2.imshow("Fruit Information", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('o'):
            fruit_name_on_screen = identify_fruit(frame=frame, confirm=False)
            local_fruit_info = get_fruit_info(fruit_name_on_screen)
            if local_fruit_info is None:
                nutrition_on_screen = "No nutrition info available."
                health_benefits_on_screen = "No health benefits info available."
            else:
                nutrition_on_screen = local_fruit_info.get("nutrition", "No nutrition info available.")
                health_benefits_on_screen = local_fruit_info.get("health_benefits", "No health benefits info available.")
        elif key == ord('s'):
            audio_file = record_audio_pyaudio(duration=3)
            recognized = recognize_speech_with_wit(audio_file, access_token)
            if recognized:
                voice_command = recognized
                print("Wit.ai 識別的語音內容：", voice_command)
            else:
                voice_command = "No voice command detected."
        elif key == ord('c'):
            print(f"\nVoice Chat Mode about {fruit_name_on_screen}:")
            voice_chat(access_token, fruit_name_on_screen, local_fruit_info)
        elif key == ord('x'):
            # 綜合操作：利用當前 frame 執行水果辨識、錄音識別與資訊查詢
            combined_operation_with_frame(frame, access_token)
    cap.release()
    cv2.destroyAllWindows()

def run_cli_mode():
    """
    CLI 模式：使用者輸入圖片路徑進行水果辨識，
    之後進入對話迴圈，可使用 change_image 指令切換圖片，
    或 new_image 開始新對話。
    """
    global fruit_name, fruit_info
    while True:
        image_path = input("\n📸 請輸入圖片路徑（或輸入 exit 結束）：").strip()
        if image_path.lower() == "exit":
            print("👋 再見！")
            break

        fruit_name = identify_fruit(image_path=image_path, confirm=True)
        fruit_info = get_fruit_info(fruit_name)
        if not fruit_info:
            print(f"⚠️ 找不到 '{fruit_name}' 的相關資訊，請自行搜尋。")
        else:
            display_fruit_info(fruit_info)

        print("\n💬 **AI 對話開始**")
        print("可輸入指令：help 查看建議問題，change_image [圖片路徑] 切換圖片，")
        print("或輸入 new_image 以使用新圖片開始對話，或輸入 exit 結束。")
        while True:
            user_input = input("\n🗨️ You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 謝謝使用，期待下次見！")
                sys.exit(0)
            elif user_input.lower().startswith("change_image"):
                new_image_path = user_input.replace("change_image", "").strip()
                change_image(new_image_path)
            elif user_input.lower() == "new_image":
                break
            elif "calories" in user_input.lower() or "卡路里" in user_input:
                answer = query_ai_for_fruit(fruit_name, fruit_info, query_type="calories")
                translated_answer = translate_to_zh(answer)
                print(f"🤖 AI: {translated_answer}")
            elif "vitamin" in user_input.lower() or "維生素" in user_input:
                answer = query_ai_for_fruit(fruit_name, fruit_info, query_type="vitamins")
                translated_answer = translate_to_zh(answer)
                print(f"🤖 AI: {translated_answer}")
            elif "health" in user_input.lower() or "益處" in user_input:
                answer = query_ai_for_fruit(fruit_name, fruit_info, query_type="health_benefits")
                translated_answer = translate_to_zh(answer)
                print(f"🤖 AI: {translated_answer}")
            elif user_input.lower() == "help":
                print("建議提問：'calories', 'vitamins', 'health benefits' 或其他通用查詢。")
            else:
                answer = query_ai_for_fruit(fruit_name, fruit_info, question=user_input)
                translated_answer = translate_to_zh(answer)
                print(f"🤖 AI: {translated_answer}")

def main():
    """
    主選單：選擇網路攝影機模式或 CLI 模式
    """
    print("Welcome to the Fruit Information System!")
    print("請選擇模式：")
    print("1: 網路攝影機模式")
    print("2: 圖片檔案 (CLI) 模式")
    mode = input("請輸入模式 (1 或 2)：").strip()

    if mode == "1":
        run_webcam_mode()
    elif mode == "2":
        run_cli_mode()
    else:
        print("模式錯誤，程式結束。")

if __name__ == "__main__":
    main()
