# Educational Animation Video Generator

## Overview
This project is a Text-Driven Educational Animation Video Generator that transforms mathematical and geometric concepts into animated educational videos with voice narration.

## Recent Changes
- November 27, 2025: Added new animation types (vector addition, bubble sort, graph extrema)
- November 27, 2025: Added scene breakdown display in frontend
- November 27, 2025: Fixed TTS to use espeak directly via subprocess
- November 27, 2025: Enhanced frontend with code preview and timeline view
- November 27, 2025: Improved error handling with user-friendly messages
- November 27, 2025: Initial project setup with full pipeline

## Project Architecture

### Backend (main.py)
- **FastAPI** web server on port 5000
- **OpenAI Integration** for generating educational explanations and animation scripts
- **Manim** for rendering mathematical animations (LaTeX-free)
- **espeak** for text-to-speech audio generation
- **ffmpeg** for video-audio merging
- **Scene Breakdown Generator** for animation timeline

### Frontend (static/index.html)
- Single-page application with modern gradient design
- Input field for concept entry
- Example buttons: Circle, Triangle, Pythagorean Theorem, Sine Wave, Vector Addition, Bubble Sort, Graph Extrema, Derivative
- Progress indicator with stage descriptions
- Video player with download option
- Scene timeline display
- Generated code preview toggle
- Feature cards highlighting capabilities

### Pipeline Flow
1. User enters a mathematical concept
2. System generates educational explanation (OpenAI or fallback)
3. System creates animation script parameters
4. Scene breakdown is generated for timeline display
5. Manim Python code is dynamically generated
6. Animation is rendered to MP4
7. Text-to-speech generates narration audio (espeak)
8. Video and audio are merged with ffmpeg
9. Final video with scene breakdown is served to user

### Supported Animation Types
- **Geometric shapes**: circle, triangle, square, rectangle
- **Theorems**: Pythagorean theorem
- **Functions**: sine wave, cosine wave, parabola, function graphs
- **Calculus**: derivative, integral, graph extrema (maxima/minima)
- **Vectors**: vector addition with parallelogram visualization
- **Algorithms**: bubble sort with step-by-step animation

### Directory Structure
```
/
├── main.py              # FastAPI backend with full pipeline
├── static/
│   └── index.html       # Frontend UI with scene breakdown
├── output_videos/       # Generated final videos
├── temp_renders/        # Temporary rendering files
└── replit.md            # Project documentation
```

## User Preferences
- No placeholders or TODOs
- Fully functional end-to-end implementation
- Single-command startup
- Clean, educational theme
- Display scene breakdowns and animation instructions

## Environment Variables
- OPENAI_API_KEY: For AI-generated explanations (optional, has fallbacks)

## System Dependencies
- espeak: For text-to-speech audio generation
- ffmpeg: For video-audio merging
