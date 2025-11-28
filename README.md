EduAnimMaker – AI Powered Educational Animation Generator

EduAnimMaker is an intelligent animation-generation tool designed to help students and teachers create short educational videos and visual explanations using Manim, Flask, and AI-generated descriptions.
It takes a topic or text input and automatically generates animated scenes that can be used for teaching, presentations, and learning.

🚀 Features
✅ AI-Generated Animation Content

Automatically converts an input text into a Manim animation script.

✅ Manim-Based Video Rendering

Uses ManimCE to generate high-quality educational animations.

✅ Web Interface (Flask)

Simple UI where users enter a topic and download the generated video.

✅ Modular Architecture

Easy to modify, extend, or integrate new models or animation styles.

📂 Project Structure
EduAnimMaker/
│── main.py              # Flask backend + Manim integration
│── templates/           # HTML frontend pages
│── static/              # CSS / JS files (if used)
│── OUTPUT/              # Generated animation videos
│── requirements.txt     # Install dependencies
│── .gitignore

🛠️ Installation
1️⃣ Clone the repository
git clone https://github.com/Umamahesh2005/EduAnimMaker.git
cd EduAnimMaker

2️⃣ Create virtual environment (optional)
python -m venv .venv
.\.venv\Scripts\activate

3️⃣ Install required libraries
pip install -r requirements.txt

4️⃣ Run the Flask app
python main.py


The application will start on:

http://127.0.0.1:5000

🎞️ How It Works

User enters a text/topic

AI generates an animation description

Script is converted into Manim Python code

Manim renders and returns the video file

User downloads the animation
