import json
import os
import ollama
import re
import sys
from difflib import get_close_matches
import wikipedia  # 載入 wikipedia 套件

# **水果資料庫**（水果名稱保持英文）
FRUIT_JSON_PATH = "/ollama_host/fruit_dataset.json"

# **紀錄使用者問過的問題，防止重複回答**
question_history = {}

def identify_fruit(image_path):
    """辨識水果名稱（保持英文），並嘗試從回應中提取出答案部分"""
    if not os.path.exists(image_path):
        print(f"❌ Cannot find image `{image_path}`. Please check the path.")
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
            "images": [image_path]
        }]
    )

    # 取得完整回應
    full_response = response["message"]["content"].strip()
    # 嘗試從回應中找出 "**Answer:**" 之後的單詞
    match = re.search(r"\*\*Answer:\*\*\s*(\w+)", full_response)
    if match:
        fruit_name = match.group(1)
    else:
        fruit_name = full_response

    # 請使用者確認辨識結果
    user_confirm = input(f"🔍 Model recognized: `{fruit_name}`. Is this correct? (yes/no): ").strip().lower()
    if user_confirm != "yes":
        fruit_name = input("Please enter the correct fruit name: ").strip().title()

    # 去除非英文字元（保留英文）
    fruit_name = re.sub(r"[^A-Za-z ]", "", fruit_name).strip().title()
    return fruit_name

def fetch_fruit_info_online(fruit_name):
    """
    使用 wikipedia 套件從線上取得該水果的資訊，盡量提供營養相關內容。
    若搜尋到的頁面摘要不足，嘗試從完整頁面中擷取 Nutrition 部分。
    對於 "Pear" 等易混淆的水果，使用 "Pear (fruit)" 進行查詢。
    """
    try:
        wikipedia.set_lang("en")
        query_name = fruit_name
        if fruit_name.lower() == "pear":
            query_name = "Pear (fruit)"
        # 取得摘要，句數多一些以提高包含 nutrition 內容的機率
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
    """從 JSON 中獲取水果資訊，若找不到則嘗試線上查詢"""
    if not os.path.exists(FRUIT_JSON_PATH):
        print(f"❌ Cannot find `{FRUIT_JSON_PATH}`. Please check the path.")
        sys.exit(1)

    with open(FRUIT_JSON_PATH, "r", encoding="utf-8") as file:
        fruit_data = json.load(file)

    # 直接比對水果名稱（保持英文）
    fruit_info = next((fruit for fruit in fruit_data if fruit_name.lower() == fruit["fruit"].lower()), None)

    if fruit_info:
        return fruit_info

    # 若找不到，進行模糊比對
    fruit_names = [fruit["fruit"] for fruit in fruit_data]
    matches = get_close_matches(fruit_name, fruit_names, n=1, cutoff=0.6)
    if matches:
        user_confirm = input(f"The fruit '{fruit_name}' is not in the database. Did you mean '{matches[0]}'? (yes/no): ").strip().lower()
        if user_confirm == "yes":
            return next((fruit for fruit in fruit_data if fruit["fruit"].lower() == matches[0].lower()), None)
    
    # 如果 JSON 中沒有找到，則嘗試從 Wikipedia 上查詢
    print(f"⚠️ No information available for '{fruit_name}' in the database. Searching Wikipedia...")
    wiki_summary = fetch_fruit_info_online(fruit_name)
    if wiki_summary:
        print("\n✅ Wikipedia returned the following info:")
        print(wiki_summary)
        user_confirm = input("Would you like to use this information? (yes/no): ").strip().lower()
        if user_confirm == "yes":
            return {
                "fruit": fruit_name,
                "nutrition": wiki_summary,
                "health_benefits": wiki_summary,
            }
    else:
        print("⚠️ Unable to retrieve valid info from Wikipedia.")
    return None

def query_ai_for_fruit(fruit_name, fruit_info, query_type="general"):
    """
    根據使用者的問題類型，利用 fruit_info 中的資訊回答：
      - 若資訊為結構化（例如 JSON 中 "Per 100g:" 開頭的資訊），則直接解析出關鍵數據。
      - 若資訊來自線上（非結構化），則利用正則表達式從中擷取營養相關內容。
    """
    if fruit_name not in question_history:
        question_history[fruit_name] = set()
    
    if query_type in question_history[fruit_name]:
        return "🤖 AI: You already asked that. Please try a different question."
    question_history[fruit_name].add(query_type)
    
    structured = "Per 100g:" in fruit_info['nutrition']
    
    if query_type == "calories":
        if structured:
            try:
                # 從 "Per 100g: 52 calories, ..." 擷取卡路里
                calories = fruit_info['nutrition'].split(':')[1].split(',')[0].strip()
                return f"{fruit_name} per 100g contains {calories} calories."
            except Exception:
                return "⚠️ Unable to parse calorie information."
        else:
            match = re.search(r'(\d+)\s*kilocalories', fruit_info['nutrition'], re.IGNORECASE)
            if match:
                cal = match.group(1)
                return f"{fruit_name} per 100g contains {cal} kilocalories."
            else:
                return f"Unable to find calorie information for {fruit_name}."
    
    elif query_type == "vitamins":
        if structured:
            try:
                # 從 "rich in ..." 擷取維生素資訊
                match = re.search(r"rich in (.+)", fruit_info['nutrition'], re.IGNORECASE)
                if match:
                    vitamins = match.group(1).strip()
                    return f"{fruit_name} is rich in {vitamins}."
                else:
                    return f"No vitamin information found for {fruit_name}."
            except Exception:
                return "⚠️ Unable to parse vitamin information."
        else:
            matches = re.findall(r'vitamin\s*([A-Za-z]+)', fruit_info['nutrition'], re.IGNORECASE)
            if matches:
                vitamins = ", ".join(sorted(set(matches)))
                return f"{fruit_name} is rich in vitamins: {vitamins}."
            else:
                return f"No vitamin information found for {fruit_name}."
    
    elif query_type == "health_benefits":
        if structured:
            return f"Health benefits of {fruit_name}: {fruit_info['health_benefits']}"
        else:
            idx = fruit_info['nutrition'].find("Research")
            if idx != -1:
                health_text = fruit_info['nutrition'][idx:]
                return f"Health benefits of {fruit_name}: {health_text}"
            else:
                return f"Health benefits of {fruit_name}: {fruit_info['nutrition']}"
    
    else:
        return f"{fruit_name} is a nutrient-rich fruit. What specific information do you need?"

def display_fruit_info(fruit_info):
    """顯示水果資訊"""
    if not fruit_info:
        print("⚠️ No fruit information available.")
        return

    print(f"\n🍎 Fruit Information:")
    print(f"🔹 Name: {fruit_info['fruit']}")
    print(f"🔹 Nutrition: {fruit_info['nutrition']}")
    print(f"🔹 Health Benefits: {fruit_info['health_benefits']}")

def change_image(new_image_path):
    """
    處理 change_image 指令，若成功切換圖片則更新全域變數，
    並在內層對話中回覆換圖成功，**但不跳出內層迴圈**。
    """
    global fruit_name, fruit_info
    if os.path.exists(new_image_path):
        fruit_name = identify_fruit(new_image_path)
        fruit_info = get_fruit_info(fruit_name)
        display_fruit_info(fruit_info)
        question_history[fruit_name] = set()
        print("\n✅ Fruit switched. You can now ask questions about the new fruit!")
        return True
    else:
        print(f"❌ Cannot find image `{new_image_path}`. Please check the path.")
        return False

# **主程式：外層迴圈**
while True:
    image_path = input("\n📸 Enter image path (or type `exit` to quit): ").strip()
    if image_path.lower() == "exit":
        print("👋 Goodbye!")
        break

    # 初始圖片辨識與資訊取得
    fruit_name = identify_fruit(image_path)
    fruit_info = get_fruit_info(fruit_name)

    if not fruit_info:
        print(f"⚠️ No information available for '{fruit_name}'. Please manually search for its info.")
    else:
        display_fruit_info(fruit_info)

    # 內層對話迴圈：持續與使用者對話
    print("\n💬 **AI Conversation Started**")
    print("Type `help` for suggested questions, type `change_image [image path]` to switch image within this conversation,")
    print("or type `new_image` to start a new conversation with a new image, or type `exit` to quit.")

    while True:
        user_input = input("\n🗨️ You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Thank you for using. See you next time!")
            sys.exit(0)
        elif user_input.lower().startswith("change_image"):
            # 直接在內層對話中換圖片
            new_image_path = user_input.replace("change_image", "").strip()
            change_image(new_image_path)
        elif user_input.lower() == "new_image":
            # 結束內層對話，回到外層重新輸入圖片路徑
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
        else:
            response = query_ai_for_fruit(fruit_name, fruit_info)
            print(f"🤖 AI: {response}")

    # 走到這裡表示使用者輸入了 "new_image"，回到外層迴圈重新輸入圖片路徑
