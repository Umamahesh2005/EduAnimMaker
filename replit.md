# Educational Animation Video Generator

## Overview
This project is a Text-Driven Educational Animation Video Generator that transforms mathematical and geometric concepts into animated educational videos with voice narration.

## Recent Changes
- Initial project setup (November 27, 2025)
- Complete backend implementation with FastAPI
- Manim animation engine integration
- Text-to-speech with pyttsx3
- Video-audio merging with ffmpeg
- HTML frontend with responsive design

## Project Architecture

### Backend (main.py)
- **FastAPI** web server on port 5000
- **OpenAI Integration** for generating educational explanations and animation scripts
- **Manim** for rendering mathematical animations
- **pyttsx3** for text-to-speech audio generation
- **ffmpeg** for video-audio merging

### Frontend (static/index.html)
- Single-page application
- Input field for concept entry
- Example buttons for quick testing
- Progress indicator during generation
- Video player with download option
- Responsive design with gradient theme

### Pipeline Flow
1. User enters a mathematical concept
2. System generates educational explanation (OpenAI or fallback)
3. System creates animation script parameters
4. Manim Python code is dynamically generated
5. Animation is rendered to MP4
6. Text-to-speech generates narration audio
7. Video and audio are merged with ffmpeg
8. Final video is served to user

### Supported Concepts
- Geometric shapes: circle, triangle, square, rectangle
- Theorems: Pythagorean theorem
- Functions: sine wave, cosine wave, parabola
- Calculus: derivative, integral

### Directory Structure
```
/
├── main.py              # FastAPI backend
├── static/
│   └── index.html       # Frontend UI
├── output_videos/       # Generated videos
├── temp_renders/        # Temporary rendering files
└── requirements.txt     # Python dependencies
```

## User Preferences
- No placeholders or TODOs
- Fully functional end-to-end implementation
- Single-command startup
- Clean, educational theme

## Environment Variables
- OPENAI_API_KEY: For AI-generated explanations (optional, has fallbacks)
