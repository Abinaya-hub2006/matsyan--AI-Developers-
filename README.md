🐟 Fish Recognition & Wikipedia Chatbot Web App
A deep learning + machine learning-powered web app that classifies fish species from images and allows users to ask fish-related questions via a chatbot. Built using PyTorch, SVM, and LangChain + Wikipedia API, and deployed with Gradio on Hugging Face Spaces.

**❓ Problem Statement**
Aquatic biodiversity is under-researched in many regions, especially among freshwater species. Identifying fish accurately from images can aid in biodiversity studies, aquaculture, and environmental monitoring.

✅ Automatic classification of 30+ common fish species from image uploads.

💬 Providing knowledge-based answers about fish via a Wikipedia-integrated chatbot.

**Dataset**
link:https://www.kaggle.com/datasets/markdaniellampa/fish-dataset


| Layer        | Tools & Libraries                                    |
| ------------ | ---------------------------------------------------- |
| 💻 Frontend  | html,css (image upload, chat interface)             |
| 🧠 Backend   | Flask (for legacy), Gradio for Spaces deployment     |
| 🐟 ML Model  | ResNet18 (feature extraction) + SVM (classification) |
| 🤖 Chatbot   | LangChain + WikipediaAPIWrapper                      |
| ☁ Deployment | Hugging Face Spaces                                  |
| 🔢 Language  | Python (3.8+)                                        |

** Installation & Setup Instructions**
⚠️ You must have Python 3.8+ installed.

🔧 1. Clone the Repository
git clone https://huggingface.co/spaces/abhinaya2006/Fish_recognition
cd Fish_recognition

🐍 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**🔍 Features**
🎯 Classify 30+ types of fish species from images.
🧠 Uses ResNet18 for deep feature extraction.
📊 Classifies with SVM for faster inference.
🤖 Built-in chatbot powered by Wikipedia + LangChain.
🧪 Fully functional offline & online (Gradio and Flask versions).
💻 No Docker needed (Gradio auto-deployment to HF Spaces).


**📝 Future Improvements**
✅ Model accuracy improvements with CNN fine-tuning
🌐 Add multilingual support for fish names
📱 Build mobile-friendly UI
💬 Integrate OpenAI or LLaMA chatbot fallback
🐠 Add endangered species tagging

<img width="1600" height="782" alt="image" src="https://github.com/user-attachments/assets/6c99008d-7fdb-49c5-ab41-52209654d76b" />

**youtube demo link:**
https://youtu.be/pvH49V3-fUA?si=jDvrtsR9z3-hV0FA



**Team details:**
AI Developers
Team members:
Abinaya K
Sharmila S

