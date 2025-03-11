import cv2
import ollama
import re
import os
import json
import sys
import wikipedia
from difflib import get_close_matches
# 水果資料庫的 JSON 路徑（注意檔名副檔名是否正確）
FRUIT_JSON_PATH = "/opt/NanoLLM/fruit_dataset.jason"
# 紀錄使用者問過的問題，避免重複查詢（可依需求使用）
question_history = {}
def identify_fruit(frame):
    """
    使用 Ollama 辨識水果 (保持英文)：
      - 先將當前畫面存成圖片
      - 呼叫 Ollama API 取得回應後解析出水果名稱
    """
    image_path = "current_frame.jpg"
    cv2.imwrite(image_path, frame)
    llava_prompt = """
    Please analyze this image and output only a single fruit name (for example, "Apple", "Banana", "Grape", "Kiwi", "Mango", "Orange", "Strawberry", "Chickoo", "Cherry").
    Only respond with the fruit name without any extra characters, punctuation, numbers, or explanation. If unsure, try to guess a similar fruit name.
    """
    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": llava_prompt,
            "images": [image_path]
        }]
    )
    # 取得回應文字，並利用正則表達式解析出水果名稱
    full_response = response["message"]["content"].strip()
    match = re.search(r"\*\*Answer:\*\*\s*(\w+)", full_response)
    if match:
        fruit_name = match.group(1)
    else:
        fruit_name = full_response
    # 去除非英文字元，並將字串格式化
    fruit_name = re.sub(r"[^A-Za-z ]", "", fruit_name).strip().title()
    return fruit_name
def fetch_fruit_info_online(fruit_name):
    """
    使用 Wikipedia 套件取得水果的線上資訊，盡量涵蓋營養資訊。
    如果摘要中沒有 Nutrition 部分，則嘗試擷取完整頁面的部分內容。
    """
    try:
        wikipedia.set_lang("en")
        query_name = fruit_name
        if fruit_name.lower() == "pear":
            query_name = "Pear (fruit)"
        summary = wikipedia.summary(query_name, sentences=5)
        if "Nutrition" not in summary:
            page = wikipedia.page(query_name)
            content = page.content
            idx = content.find("Nutrition")
            if idx != -1:
                nutrition_excerpt = content[idx:idx+500]
                summary += "\n\nNutrition Info:\n" + nutrition_excerpt
        return summary
    except Exception as e:
        print(f":警告: Error fetching info from Wikipedia: {e}")
        return "No nutrition information available."
def get_fruit_info(fruit_name):
    """
    從本地 JSON 資料庫取得水果資訊，如果找不到則嘗試線上查詢。
    回傳的 fruit_info 應為一個 dictionary，包含至少 'fruit' 與 'nutrition' 兩個 key。
    """
    if not os.path.exists(FRUIT_JSON_PATH):
        print(f":x: Cannot find `{FRUIT_JSON_PATH}`. Please check the path.")
        sys.exit(1)
    with open(FRUIT_JSON_PATH, "r", encoding="utf-8") as file:
        fruit_data = json.load(file)
    # 直接比對水果名稱（忽略大小寫）
    fruit_info = next((fruit for fruit in fruit_data if fruit_name.lower() == fruit["fruit"].lower()), None)
    if fruit_info:
        return fruit_info
    # 模糊比對找出相近的名稱
    fruit_names = [fruit["fruit"] for fruit in fruit_data]
    matches = get_close_matches(fruit_name, fruit_names, n=1, cutoff=0.6)
    if matches:
        user_confirm = input(f"The fruit '{fruit_name}' is not in the database. Did you mean '{matches[0]}'? (yes/no): ").strip().lower()
        if user_confirm == "yes":
            return next((fruit for fruit in fruit_data if fruit["fruit"].lower() == matches[0].lower()), None)
    # 如果資料庫中沒有，則改從 Wikipedia 取得資訊
    print(f":警告: No information available for '{fruit_name}' in the database. Searching Wikipedia...")
    wiki_summary = fetch_fruit_info_online(fruit_name)
    return {
        "fruit": fruit_name,
        "nutrition": wiki_summary
    }
def main():
    cap = cv2.VideoCapture(4)  # 根據需要設定攝影機編號
    if not cap.isOpened():
        print("無法開啟攝影機")
        sys.exit(1)
    print("按下 'o' 鍵進行水果辨識，按 'q' 鍵退出。")
    # 儲存要顯示在影像上的文字，預設為空
    fruit_name_on_screen = ""
    nutrition_on_screen = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法擷取影像")
            break
        # 將水果名稱與營養資訊疊加到影像上
        # 例如：水果名稱顯示在 (10, 40)，營養資訊從 (10, 80) 開始，並每行間隔 30 像素
        cv2.putText(frame, f"Fruit: {fruit_name_on_screen}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # 將 nutrition_on_screen 分割成多行，避免文字太長
        for i, line in enumerate(nutrition_on_screen.splitlines()):
            y_pos = 80 + i * 30
            cv2.putText(frame, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Video Output", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            # 呼叫辨識函式
            fruit_name = identify_fruit(frame)
            fruit_name_on_screen = fruit_name  # 記錄辨識到的水果名稱
            # 從資料庫或線上取得該水果的營養資訊
            fruit_info = get_fruit_info(fruit_name)
            nutrition_on_screen = fruit_info.get("nutrition", "")
            print(f"辨識到的水果：{fruit_name}")
            print("營養資訊：")
            print(nutrition_on_screen)
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()