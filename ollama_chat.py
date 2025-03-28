import cv2
import ollama
import re
import os
import json
import sys
import wikipedia
from difflib import get_close_matches

# 統一水果資料庫的 JSON 檔案路徑
FRUIT_JSON_PATH = "/opt/NanoLLM/ollama_host/fruit_dataset.json"

# 記錄使用者問過的問題，防止重複回答
question_history = {}

# 全域變數，方便在 CLI 模式下更換圖片時更新水果資訊
fruit_name = ""
fruit_info = {}

def identify_fruit(frame=None, image_path=None, confirm=True):
    """
    辨識水果名稱，輸入可以是攝影機擷取的 frame 或圖片路徑。
    若 confirm 為 True 則會請使用者確認辨識結果（CLI 模式）。
    """
    if frame is not None:
        temp_path = "current_frame.jpg"
        cv2.imwrite(temp_path, frame)
        image_source = temp_path
    elif image_path is not None:
        if not os.path.exists(image_path):
            print(f"❌ Cannot find image `{image_path}`. Please check the path.")
            return None
        image_source = image_path
    else:
        print("❌ No image source provided.")
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
    # 如果回應中包含 "**Answer:**"，則嘗試擷取其後的單字
    match = re.search(r"\*\*Answer:\*\*\s*(\w+)", fruit_result)
    if match:
        recognized = match.group(1)
    else:
        recognized = fruit_result

    recognized = recognized.title()
    recognized = re.sub(r"[^A-Za-z ]", "", recognized).strip()

    if confirm:
        user_confirm = input(f"🔍 Model recognized: `{recognized}`. Is this correct? (yes/no): ").strip().lower()
        if user_confirm != "yes":
            recognized = input("Please enter the correct fruit name: ").strip().title()
            recognized = re.sub(r"[^A-Za-z ]", "", recognized).strip()

    return recognized

def fetch_fruit_info_online(fruit_name):
    """
    使用 wikipedia 套件從線上取得該水果的資訊，盡量提供營養相關內容。
    若摘要中 Nutrition 內容不足，嘗試從完整頁面中擷取部分 Nutrition 資訊。
    對於 "Pear" 等易混淆的水果，使用 "Pear (fruit)" 進行查詢。
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
    except wikipedia.DisambiguationError as e:
        print(f"⚠️ Multiple results found: {e.options}")
        return None
    except Exception as e:
        print(f"⚠️ Error fetching info from Wikipedia: {e}")
        return None

def get_fruit_info(fruit_name):
    """
    從 JSON 資料庫中獲取水果資訊，若找不到則嘗試線上查詢。
    若進行模糊比對後有候選結果，請使用者確認。
    """
    if not os.path.exists(FRUIT_JSON_PATH):
        print(f"❌ Cannot find `{FRUIT_JSON_PATH}`. Please check the path.")
        sys.exit(1)

    with open(FRUIT_JSON_PATH, "r", encoding="utf-8") as file:
        fruit_data = json.load(file)

    # 嘗試直接比對水果名稱（保持英文）
    info = next((fruit for fruit in fruit_data if fruit_name.lower() == fruit["fruit"].lower()), None)
    if info:
        return info

    # 若找不到，進行模糊比對
    fruit_names = [fruit["fruit"] for fruit in fruit_data]
    matches = get_close_matches(fruit_name, fruit_names, n=1, cutoff=0.6)
    if matches:
        user_confirm = input(f"The fruit '{fruit_name}' is not in the database. Did you mean '{matches[0]}'? (yes/no): ").strip().lower()
        if user_confirm == "yes":
            return next((fruit for fruit in fruit_data if fruit["fruit"].lower() == matches[0].lower()), None)

    print(f"⚠️ No information available for '{fruit_name}' in the database. Searching Wikipedia...")
    wiki_summary = fetch_fruit_info_online(fruit_name)
    if wiki_summary:
        user_confirm = input("Would you like to use this information? (yes/no): ").strip().lower()
        if user_confirm == "yes":
            return {
                "fruit": fruit_name,
                "nutrition": wiki_summary,
                "health_benefits": wiki_summary,
            }
    return None

def query_ai_for_fruit(fruit_name, fruit_info, query_type="general"):
    """
    根據使用者詢問的問題類型，從 fruit_info 中解析資訊回答：
      - calories：解析每 100g 的卡路里資訊
      - vitamins：解析水果所含維生素資訊
      - health_benefits：回傳健康益處相關內容
      - general：給出通用回應
    """
    if fruit_name not in question_history:
        question_history[fruit_name] = set()
    
    if query_type in question_history[fruit_name]:
        return "🤖 AI: You already asked that. Please try a different question."
    question_history[fruit_name].add(query_type)
    
    structured = "Per 100g:" in fruit_info.get('nutrition', "")
    
    if query_type == "calories":
        if structured:
            try:
                calories = fruit_info['nutrition'].split(':')[1].split(',')[0].strip()
                return f"{fruit_name} per 100g contains {calories} calories."
            except Exception:
                return "⚠️ Unable to parse calorie information."
        else:
            match = re.search(r'(\d+)\s*kilocalories', fruit_info.get('nutrition', ""), re.IGNORECASE)
            if match:
                cal = match.group(1)
                return f"{fruit_name} per 100g contains {cal} kilocalories."
            else:
                return f"Unable to find calorie information for {fruit_name}."
    
    elif query_type == "vitamins":
        if structured:
            try:
                match = re.search(r"rich in (.+)", fruit_info.get('nutrition', ""), re.IGNORECASE)
                if match:
                    vitamins = match.group(1).strip()
                    return f"{fruit_name} is rich in {vitamins}."
                else:
                    return f"No vitamin information found for {fruit_name}."
            except Exception:
                return "⚠️ Unable to parse vitamin information."
        else:
            matches = re.findall(r'vitamin\s*([A-Za-z]+)', fruit_info.get('nutrition', ""), re.IGNORECASE)
            if matches:
                vitamins = ", ".join(sorted(set(matches)))
                return f"{fruit_name} is rich in vitamins: {vitamins}."
            else:
                return f"No vitamin information found for {fruit_name}."
    
    elif query_type == "health_benefits":
        if structured:
            return f"Health benefits of {fruit_name}: {fruit_info.get('health_benefits', 'No health benefits info.')}"
        else:
            idx = fruit_info.get('nutrition', "").find("Research")
            if idx != -1:
                health_text = fruit_info.get('nutrition', "")[idx:]
                return f"Health benefits of {fruit_name}: {health_text}"
            else:
                return f"Health benefits of {fruit_name}: {fruit_info.get('nutrition', '')}"
    
    else:
        return f"{fruit_name} is a nutrient-rich fruit. What specific information do you need?"

def display_fruit_info(fruit_info):
    """在 CLI 模式下將水果資訊印出來"""
    if not fruit_info:
        print("⚠️ No fruit information available.")
        return
    print(f"\n🍎 Fruit Information:")
    print(f"🔹 Name: {fruit_info.get('fruit', 'Unknown')}")
    print(f"🔹 Nutrition: {fruit_info.get('nutrition', 'N/A')}")
    print(f"🔹 Health Benefits: {fruit_info.get('health_benefits', 'N/A')}")

def change_image(new_image_path):
    """
    在 CLI 對話中切換圖片，更新全域變數 fruit_name 與 fruit_info，
    並重置該水果的問答紀錄。
    """
    global fruit_name, fruit_info
    if os.path.exists(new_image_path):
        fruit_name = identify_fruit(image_path=new_image_path, confirm=True)
        fruit_info = get_fruit_info(fruit_name)
        display_fruit_info(fruit_info)
        question_history[fruit_name] = set()
        print("\n✅ Fruit switched. You can now ask questions about the new fruit!")
        return True
    else:
        print(f"❌ Cannot find image `{new_image_path}`. Please check the path.")
        return False

def wrap_text(text, font, font_scale, thickness, max_width):
    """
    Wrap text into multiple lines based on the maximum width.
    """
    lines = []
    words = text.split(' ')
    current_line = ""
    
    for word in words:
        # Combine the current line with the next word
        test_line = current_line + " " + word if current_line else word
        # Get the size of the test line
        (width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if width <= max_width:
            # If the test line fits, update the current line
            current_line = test_line
        else:
            # If it doesn't fit, add the current line to lines and start a new one
            lines.append(current_line)
            current_line = word  # Start a new line with the current word
            
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    return lines

def run_webcam_mode():
    """
    網路攝影機模式：使用 OpenCV 擷取即時影像，
    按下 'o' 進行水果辨識，'c' 進入對話模式，'q' 離開程式。
    """
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fruit_name_on_screen = ""
    nutrition_on_screen = ""
    health_benefits_on_screen = ""
    local_fruit_info = {}

    print("Press 'o' to recognize fruit, 'c' to chat, 'q' to quit.")
    
    # Create the window and set it to a specific size and position
    cv2.namedWindow("Fruit Information", cv2.WINDOW_NORMAL)  # Allow resizing
    cv2.resizeWindow("Fruit Information", 1000, 600)  # Set window size
    cv2.moveWindow("Fruit Information", 100, 200)  # Set window position

    # Max width for text
    max_width = 1800
    fruit_name_y_pos = 50  # Fixed position for fruit name
    y_pos = 150  # Initial position for nutrition information

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Wrap the fruit name (this part remains unchanged)
        fruit_lines = wrap_text(f"Fruit: {fruit_name_on_screen}", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2, max_width)
        for line in fruit_lines:
            cv2.putText(frame, line, (10, fruit_name_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            # No need to update fruit_name_y_pos, keeping it fixed

        # Position for nutrition info (fixed position)
        nutrition_y_pos = y_pos  # Fixed position for nutrition info
        # Wrap nutrition information
        nutrition_lines = wrap_text(nutrition_on_screen, cv2.FONT_HERSHEY_SIMPLEX, 1, 1, max_width)
        for line in nutrition_lines:
            cv2.putText(frame, line, (10, nutrition_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            nutrition_y_pos += 50

        # Position for health benefits info (fixed position)
        health_benefits_y_pos = nutrition_y_pos + 30  # Fixed position for health benefits info
        # Wrap health benefits information
        health_benefits_lines = wrap_text(health_benefits_on_screen, cv2.FONT_HERSHEY_SIMPLEX, 1, 1, max_width)
        for line in health_benefits_lines:
            cv2.putText(frame, line, (10, health_benefits_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            health_benefits_y_pos += 50

        cv2.imshow("Fruit Information", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('o'):
            # 在攝影機模式下不進行使用者確認，避免打斷即時影像
            fruit_name_on_screen = identify_fruit(frame=frame, confirm=False)
            local_fruit_info = get_fruit_info(fruit_name_on_screen)
            if local_fruit_info is None:
                nutrition_on_screen = "No nutrition info available."
                health_benefits_on_screen = "No health benefits info available."
            else:
                nutrition_on_screen = local_fruit_info.get("nutrition", "No nutrition info available.")
                health_benefits_on_screen = local_fruit_info.get("health_benefits", "No health benefits info available.")
        elif key == ord('c'):
            print(f"\nChatting about {fruit_name_on_screen}:")
            while True:
                user_input = input("🗨️ You (type 'exit' to go back): ").lower()
                if user_input in ["exit", "quit", "back"]:
                    break
                elif "calories" in user_input:
                    print(query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="calories"))
                elif "vitamin" in user_input:
                    print(query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="vitamins"))
                elif "health" in user_input:
                    print(query_ai_for_fruit(fruit_name_on_screen, local_fruit_info, query_type="health_benefits"))
                else:
                    print(query_ai_for_fruit(fruit_name_on_screen, local_fruit_info))

    cap.release()
    cv2.destroyAllWindows()



def run_cli_mode():
    """
    CLI 模式：使用者依提示輸入圖片路徑進行水果辨識，之後進入對話迴圈，
    可使用 change_image 指令切換圖片，或 new_image 開始新對話。
    """
    global fruit_name, fruit_info
    while True:
        image_path = input("\n📸 Enter image path (or type `exit` to quit): ").strip()
        if image_path.lower() == "exit":
            print("👋 Goodbye!")
            break

        fruit_name = identify_fruit(image_path=image_path, confirm=True)
        fruit_info = get_fruit_info(fruit_name)

        if not fruit_info:
            print(f"⚠️ No information available for '{fruit_name}'. Please manually search for its info.")
        else:
            display_fruit_info(fruit_info)

        print("\n💬 **AI Conversation Started**")
        print("Type `help` for suggested questions, type `change_image [image path]` to switch image within this conversation,")
        print("or type `new_image` to start a new conversation with a new image, or type `exit` to quit.")

        while True:
            user_input = input("\n🗨️ You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Thank you for using. See you next time!")
                sys.exit(0)
            elif user_input.lower().startswith("change_image"):
                new_image_path = user_input.replace("change_image", "").strip()
                change_image(new_image_path)
            elif user_input.lower() == "new_image":
                break
            elif "calories" in user_input.lower() or "卡路里" in user_input:
                response = query_ai_for_fruit(fruit_name, fruit_info, query_type="calories")
                print(f"🤖 AI: {response}")
            elif "vitamin" in user_input.lower() or "維生素" in user_input:
                response = query_ai_for_fruit(fruit_name, fruit_info, query_type="vitamins")
                print(f"🤖 AI: {response}")
            elif "health" in user_input.lower() or "益處" in user_input:
                response = query_ai_for_fruit(fruit_name, fruit_info, query_type="health_benefits")
                print(f"🤖 AI: {response}")
            elif user_input.lower() == "help":
                print("Suggested questions: 'calories', 'vitamins', 'health benefits', or general inquiries.")
            else:
                response = query_ai_for_fruit(fruit_name, fruit_info)
                print(f"🤖 AI: {response}")

def main():
    """
    主選單：請選擇使用網路攝影機模式或是檔案模式
    """
    print("Welcome to the Fruit Information System!")
    print("Select mode:")
    print("1: Webcam Mode")
    print("2: Image File (CLI) Mode")
    mode = input("Enter mode (1 or 2): ").strip()

    if mode == "1":
        run_webcam_mode()
    elif mode == "2":
        run_cli_mode()
    else:
        print("Invalid mode selected. Exiting.")

if __name__ == "__main__":
    main()