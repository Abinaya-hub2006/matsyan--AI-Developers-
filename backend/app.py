{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44f7d336-d35d-4949-9d39-322de79bd791",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔍 Loading SVM model from: C:\\Users\\abina\\fish_recognition\\backend\\..\\models\\svm_model.pkl\n",
      "🔍 Loading ResNet18 feature extractor...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using cache found in C:\\Users\\abina/.cache\\torch\\hub\\pytorch_vision_v0.10.0\n",
      "C:\\Users\\abina\\anaconda3\\Lib\\site-packages\\torchvision\\models\\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.\n",
      "  warnings.warn(\n",
      "C:\\Users\\abina\\anaconda3\\Lib\\site-packages\\torchvision\\models\\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.\n",
      "  warnings.warn(msg)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔧 app.py is running...\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [17/Jul/2025 23:06:56] \"GET / HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [17/Jul/2025 23:07:07] \"POST /upload HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[🤖] Fetching from Wikipedia for: bangus\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [17/Jul/2025 23:07:14] \"POST /chat HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[🤖] Fetching from Wikipedia for: gourami\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [17/Jul/2025 23:07:30] \"POST /chat HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [17/Jul/2025 23:43:18] \"POST /upload HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[🤖] Fetching from Wikipedia for: give about bangus\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [17/Jul/2025 23:43:30] \"POST /chat HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [17/Jul/2025 23:45:32] \"GET / HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [17/Jul/2025 23:45:45] \"POST /upload HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[🤖] Fetching from Wikipedia for: tell me about Gourami fish\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [17/Jul/2025 23:45:58] \"POST /chat HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[🤖] Fetching from Wikipedia for: Gourami\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [17/Jul/2025 23:46:13] \"POST /chat HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [17/Jul/2025 23:46:42] \"POST /upload HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[🤖] Fetching from Wikipedia for: bangus Milkfish\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [17/Jul/2025 23:46:55] \"POST /chat HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import pickle\n",
    "import torch\n",
    "import torchvision.transforms as transforms\n",
    "import numpy as np\n",
    "from flask import Flask, request, jsonify, send_file\n",
    "from PIL import Image\n",
    "\n",
    "# ------------ CONFIGURATION ------------\n",
    "try:\n",
    "    BASE_DIR = os.path.abspath(os.path.dirname(__file__))\n",
    "except NameError:\n",
    "    BASE_DIR = os.getcwd()\n",
    "\n",
    "MODEL_PATH = os.path.join(BASE_DIR, \"..\", \"models\", \"svm_model.pkl\")\n",
    "FRONTEND_PATH = os.path.join(BASE_DIR, \"..\", \"frontend\", \"index.html\")\n",
    "UPLOAD_FOLDER = os.path.join(BASE_DIR, \"uploads\")\n",
    "os.makedirs(UPLOAD_FOLDER, exist_ok=True)\n",
    "\n",
    "# ------------ FLASK APP INIT ------------\n",
    "app = Flask(__name__, static_folder=\"../frontend\", template_folder=\"../frontend\")\n",
    "\n",
    "# ------------ FISH LABELS ------------\n",
    "fish_labels = {\n",
    "    0: \"Bangus\", 1: \"Big Head Carp\", 2: \"Black Spotted Barb\", 3: \"Catfish\", 4: \"Climbing Perch\",\n",
    "    5: \"Fourfinger Threadfin\", 6: \"Freshwater Eel\", 7: \"Glass Perchlet\", 8: \"Goby\", 9: \"Gold Fish\",\n",
    "    10: \"Gourami\", 11: \"Grass Carp\", 12: \"Green Spotted Puffer\", 13: \"Indian Carp\",\n",
    "    14: \"Indo-Pacific Tarpon\", 15: \"Jaguar Gapote\", 16: \"Janitor Fish\", 17: \"Knifefish\",\n",
    "    18: \"Long-Snouted Pipefish\", 19: \"Mosquito Fish\", 20: \"Mudfish\", 21: \"Mullet\",\n",
    "    22: \"Pangasius\", 23: \"Perch\", 24: \"Scat Fish\", 25: \"Silver Barb\", 26: \"Silver Carp\",\n",
    "    27: \"Silver Perch\", 28: \"Snakehead\", 29: \"Tenpounder\", 30: \"Tilapia\"\n",
    "}\n",
    "\n",
    "# ------------ LOAD MODEL ------------\n",
    "print(f\"🔍 Loading SVM model from: {MODEL_PATH}\")\n",
    "with open(MODEL_PATH, \"rb\") as f:\n",
    "    svm_model = pickle.load(f)\n",
    "\n",
    "# ------------ LOAD FEATURE EXTRACTOR ------------\n",
    "print(\"🔍 Loading ResNet18 feature extractor...\")\n",
    "resnet18 = torch.hub.load(\"pytorch/vision:v0.10.0\", \"resnet18\", pretrained=True)\n",
    "resnet18.fc = torch.nn.Identity()\n",
    "resnet18.eval()\n",
    "\n",
    "transform = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "def extract_features(image):\n",
    "    image = transform(image).unsqueeze(0)\n",
    "    with torch.no_grad():\n",
    "        features = resnet18(image).flatten().numpy()\n",
    "    return features\n",
    "\n",
    "# ------------ ROUTES ------------\n",
    "@app.route(\"/\")\n",
    "def home():\n",
    "    return send_file(FRONTEND_PATH)\n",
    "\n",
    "@app.route(\"/upload\", methods=[\"POST\"])\n",
    "def upload():\n",
    "    if \"image\" not in request.files:\n",
    "        return jsonify({\"error\": \"No image uploaded.\"}), 400\n",
    "\n",
    "    file = request.files[\"image\"]\n",
    "    if file.filename == \"\":\n",
    "        return jsonify({\"error\": \"No selected file.\"}), 400\n",
    "\n",
    "    save_path = os.path.join(UPLOAD_FOLDER, file.filename)\n",
    "    file.save(save_path)\n",
    "\n",
    "    try:\n",
    "        image = Image.open(save_path).convert(\"RGB\")\n",
    "        features = extract_features(image)\n",
    "        prediction = svm_model.predict([features])[0]\n",
    "        fish_label = fish_labels.get(prediction, \"Unknown Fish\")\n",
    "\n",
    "        return jsonify({\n",
    "            \"fish_category\": fish_label,\n",
    "            \"image_url\": f\"/uploads/{file.filename}\"\n",
    "        })\n",
    "\n",
    "    except Exception as e:\n",
    "        print(f\"[❌] Prediction error: {e}\")\n",
    "        return jsonify({\"error\": f\"Prediction failed: {str(e)}\"}), 500\n",
    "\n",
    "@app.route(\"/uploads/<filename>\")\n",
    "def uploaded_file(filename):\n",
    "    return send_file(os.path.join(UPLOAD_FOLDER, filename))\n",
    "\n",
    "# 🔧 Dummy Chat Endpoint (works offline with fallback answer)\n",
    "from langchain.chains import RetrievalQA\n",
    "from langchain_community.utilities import WikipediaAPIWrapper\n",
    "from langchain_core.prompts import PromptTemplate\n",
    "from langchain.chains.llm import LLMChain\n",
    "from langchain_community.llms import HuggingFaceHub\n",
    "\n",
    "# For fast demo, you can use HuggingFace (or OpenAI if you have key)\n",
    "from langchain_community.llms import HuggingFaceHub\n",
    "\n",
    "# ⚠️ Set your Hugging Face token (if needed)\n",
    "# os.environ[\"HUGGINGFACEHUB_API_TOKEN\"] = \"your_token_here\"\n",
    "\n",
    "# Wikipedia wrapper\n",
    "wiki = WikipediaAPIWrapper(top_k_results=1)\n",
    "\n",
    "@app.route(\"/chat\", methods=[\"POST\"])\n",
    "def chat():\n",
    "    data = request.get_json()\n",
    "    user_msg = data.get(\"message\", \"\").strip()\n",
    "\n",
    "    try:\n",
    "        if not user_msg:\n",
    "            return jsonify({\"response\": \"❌ Please ask something.\"})\n",
    "\n",
    "        print(f\"[🤖] Fetching from Wikipedia for: {user_msg}\")\n",
    "        wiki_summary = wiki.run(user_msg)\n",
    "\n",
    "        if wiki_summary:\n",
    "            return jsonify({\"response\": wiki_summary})\n",
    "        else:\n",
    "            return jsonify({\"response\": \"❌ I couldn't find anything useful.\"})\n",
    "    except Exception as e:\n",
    "        print(f\"[❌] Chatbot error: {e}\")\n",
    "        return jsonify({\"response\": \"❌ Error contacting Wikipedia bot.\"}), 500\n",
    "\n",
    "\n",
    "# ------------ MAIN ------------\n",
    "if __name__ == \"__main__\":\n",
    "    print(\"🔧 app.py is running...\")\n",
    "    app.run(debug=True, use_reloader=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "954cf836-d2be-4968-9ebe-bd2397a81132",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: langchain in c:\\users\\abina\\anaconda3\\lib\\site-packages (0.3.26)\n",
      "Collecting langchain-community\n",
      "  Downloading langchain_community-0.3.27-py3-none-any.whl.metadata (2.9 kB)\n",
      "Requirement already satisfied: wikipedia in c:\\users\\abina\\anaconda3\\lib\\site-packages (1.4.0)\n",
      "Requirement already satisfied: langchain-core<1.0.0,>=0.3.66 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain) (0.3.69)\n",
      "Requirement already satisfied: langchain-text-splitters<1.0.0,>=0.3.8 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain) (0.3.8)\n",
      "Requirement already satisfied: langsmith>=0.1.17 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain) (0.4.6)\n",
      "Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain) (2.10.6)\n",
      "Requirement already satisfied: SQLAlchemy<3,>=1.4 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain) (2.0.30)\n",
      "Requirement already satisfied: requests<3,>=2 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain) (2.32.2)\n",
      "Requirement already satisfied: PyYAML>=5.3 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain) (6.0.1)\n",
      "Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain-community) (3.9.5)\n",
      "Requirement already satisfied: tenacity!=8.4.0,<10,>=8.1.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain-community) (8.2.2)\n",
      "Collecting dataclasses-json<0.7,>=0.5.7 (from langchain-community)\n",
      "  Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)\n",
      "Collecting pydantic-settings<3.0.0,>=2.4.0 (from langchain-community)\n",
      "  Downloading pydantic_settings-2.10.1-py3-none-any.whl.metadata (3.4 kB)\n",
      "Collecting httpx-sse<1.0.0,>=0.4.0 (from langchain-community)\n",
      "  Downloading httpx_sse-0.4.1-py3-none-any.whl.metadata (9.4 kB)\n",
      "Requirement already satisfied: numpy>=1.26.2 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain-community) (1.26.4)\n",
      "Requirement already satisfied: beautifulsoup4 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from wikipedia) (4.12.3)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (1.2.0)\n",
      "Requirement already satisfied: attrs>=17.3.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (23.1.0)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (1.4.0)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (6.0.4)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (1.9.3)\n",
      "Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain-community)\n",
      "  Downloading marshmallow-3.26.1-py3-none-any.whl.metadata (7.3 kB)\n",
      "Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7,>=0.5.7->langchain-community)\n",
      "  Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)\n",
      "Requirement already satisfied: jsonpatch<2.0,>=1.33 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain-core<1.0.0,>=0.3.66->langchain) (1.33)\n",
      "Requirement already satisfied: typing-extensions>=4.7 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain-core<1.0.0,>=0.3.66->langchain) (4.12.2)\n",
      "Requirement already satisfied: packaging>=23.2 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langchain-core<1.0.0,>=0.3.66->langchain) (23.2)\n",
      "Requirement already satisfied: httpx<1,>=0.23.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langsmith>=0.1.17->langchain) (0.27.0)\n",
      "Requirement already satisfied: orjson<4.0.0,>=3.9.14 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langsmith>=0.1.17->langchain) (3.11.0)\n",
      "Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langsmith>=0.1.17->langchain) (1.0.0)\n",
      "Requirement already satisfied: zstandard<0.24.0,>=0.23.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from langsmith>=0.1.17->langchain) (0.23.0)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.6.0)\n",
      "Requirement already satisfied: pydantic-core==2.27.2 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.27.2)\n",
      "Requirement already satisfied: python-dotenv>=0.21.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from pydantic-settings<3.0.0,>=2.4.0->langchain-community) (0.21.0)\n",
      "Collecting typing-inspection>=0.4.0 (from pydantic-settings<3.0.0,>=2.4.0->langchain-community)\n",
      "  Downloading typing_inspection-0.4.1-py3-none-any.whl.metadata (2.6 kB)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from requests<3,>=2->langchain) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from requests<3,>=2->langchain) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from requests<3,>=2->langchain) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from requests<3,>=2->langchain) (2024.12.14)\n",
      "Requirement already satisfied: greenlet!=0.4.17 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from SQLAlchemy<3,>=1.4->langchain) (3.0.1)\n",
      "Requirement already satisfied: soupsieve>1.2 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from beautifulsoup4->wikipedia) (2.5)\n",
      "Requirement already satisfied: anyio in c:\\users\\abina\\anaconda3\\lib\\site-packages (from httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (4.2.0)\n",
      "Requirement already satisfied: httpcore==1.* in c:\\users\\abina\\anaconda3\\lib\\site-packages (from httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (1.0.2)\n",
      "Requirement already satisfied: sniffio in c:\\users\\abina\\anaconda3\\lib\\site-packages (from httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (1.3.0)\n",
      "Requirement already satisfied: h11<0.15,>=0.13 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (0.14.0)\n",
      "Requirement already satisfied: jsonpointer>=1.9 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from jsonpatch<2.0,>=1.33->langchain-core<1.0.0,>=0.3.66->langchain) (2.1)\n",
      "Requirement already satisfied: mypy-extensions>=0.3.0 in c:\\users\\abina\\anaconda3\\lib\\site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain-community) (1.0.0)\n",
      "Downloading langchain_community-0.3.27-py3-none-any.whl (2.5 MB)\n",
      "   ---------------------------------------- 0.0/2.5 MB ? eta -:--:--\n",
      "   ---------------------------------------- 0.0/2.5 MB ? eta -:--:--\n",
      "    --------------------------------------- 0.1/2.5 MB 825.8 kB/s eta 0:00:03\n",
      "   --- ------------------------------------ 0.2/2.5 MB 1.7 MB/s eta 0:00:02\n",
      "   --- ------------------------------------ 0.2/2.5 MB 1.7 MB/s eta 0:00:02\n",
      "   --- ------------------------------------ 0.2/2.5 MB 1.7 MB/s eta 0:00:02\n",
      "   ----- ---------------------------------- 0.3/2.5 MB 1.2 MB/s eta 0:00:02\n",
      "   ----- ---------------------------------- 0.3/2.5 MB 1.2 MB/s eta 0:00:02\n",
      "   ------- -------------------------------- 0.5/2.5 MB 1.3 MB/s eta 0:00:02\n",
      "   ---------- ----------------------------- 0.7/2.5 MB 1.6 MB/s eta 0:00:02\n",
      "   ---------- ----------------------------- 0.7/2.5 MB 1.5 MB/s eta 0:00:02\n",
      "   -------------- ------------------------- 0.9/2.5 MB 1.9 MB/s eta 0:00:01\n",
      "   ---------------- ----------------------- 1.0/2.5 MB 1.9 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 1.1/2.5 MB 1.8 MB/s eta 0:00:01\n",
      "   ------------------- -------------------- 1.2/2.5 MB 2.0 MB/s eta 0:00:01\n",
      "   -------------------- ------------------- 1.3/2.5 MB 1.8 MB/s eta 0:00:01\n",
      "   ------------------------ --------------- 1.6/2.5 MB 2.2 MB/s eta 0:00:01\n",
      "   ------------------------ --------------- 1.6/2.5 MB 2.2 MB/s eta 0:00:01\n",
      "   ---------------------------- ----------- 1.8/2.5 MB 2.2 MB/s eta 0:00:01\n",
      "   ----------------------------- ---------- 1.9/2.5 MB 2.2 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 2.0/2.5 MB 2.1 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 2.2/2.5 MB 2.2 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 2.3/2.5 MB 2.2 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 2.3/2.5 MB 2.2 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 2.5/2.5 MB 2.3 MB/s eta 0:00:00\n",
      "Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)\n",
      "Downloading httpx_sse-0.4.1-py3-none-any.whl (8.1 kB)\n",
      "Downloading pydantic_settings-2.10.1-py3-none-any.whl (45 kB)\n",
      "   ---------------------------------------- 0.0/45.2 kB ? eta -:--:--\n",
      "   ---------------------------------------- 45.2/45.2 kB 2.3 MB/s eta 0:00:00\n",
      "Downloading marshmallow-3.26.1-py3-none-any.whl (50 kB)\n",
      "   ---------------------------------------- 0.0/50.9 kB ? eta -:--:--\n",
      "   ---------------------------------------- 50.9/50.9 kB 2.5 MB/s eta 0:00:00\n",
      "Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)\n",
      "Downloading typing_inspection-0.4.1-py3-none-any.whl (14 kB)\n",
      "Installing collected packages: typing-inspection, typing-inspect, marshmallow, httpx-sse, dataclasses-json, pydantic-settings, langchain-community\n",
      "Successfully installed dataclasses-json-0.6.7 httpx-sse-0.4.1 langchain-community-0.3.27 marshmallow-3.26.1 pydantic-settings-2.10.1 typing-inspect-0.9.0 typing-inspection-0.4.1\n"
     ]
    }
   ],
   "source": [
    "!pip install langchain langchain-community wikipedia\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95c90018-9991-4e61-bbf1-da823844bd18",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
