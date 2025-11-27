import os
import json
import uuid
import subprocess
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user

app = FastAPI(title="Educational Animation Video Generator")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OUTPUT_DIR = Path("output_videos")
STATIC_DIR = Path("static")
TEMP_DIR = Path("temp_renders")

OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/videos", StaticFiles(directory="output_videos"), name="videos")


class TextInput(BaseModel):
    text: str


def generate_explanation(concept: str) -> str:
    """Generate an educational explanation for the concept using OpenAI."""
    if not OPENAI_API_KEY:
        return get_fallback_explanation(concept)
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "You are an educational content creator. Generate a clear, concise educational explanation (2-4 sentences) that would accompany an animation about the given mathematical or geometric concept. The explanation should be suitable for text-to-speech narration."
                },
                {
                    "role": "user",
                    "content": f"Create a brief educational explanation for: {concept}"
                }
            ],
            max_completion_tokens=300
        )
        content = response.choices[0].message.content
        return content if content else get_fallback_explanation(concept)
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return get_fallback_explanation(concept)


def get_fallback_explanation(concept: str) -> str:
    """Fallback explanations when OpenAI is not available."""
    concept_lower = concept.lower().strip()
    
    explanations = {
        "circle": "A circle is a perfectly round shape where every point on its edge is exactly the same distance from its center. This distance is called the radius. The circle is one of the most fundamental shapes in geometry and appears everywhere in nature and engineering.",
        "triangle": "A triangle is a polygon with three sides and three angles. The sum of all angles in any triangle always equals 180 degrees. Triangles are the strongest shape in construction because they cannot be deformed without changing the length of their sides.",
        "square": "A square is a special rectangle where all four sides are equal in length and all four angles are right angles of 90 degrees. It has both horizontal and vertical lines of symmetry, making it one of the most regular polygons.",
        "rectangle": "A rectangle is a quadrilateral with four right angles. Opposite sides are equal in length and parallel to each other. The area of a rectangle is calculated by multiplying its length by its width.",
        "pythagorean theorem": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides. Written as a squared plus b squared equals c squared, this theorem is fundamental to geometry and has countless real-world applications.",
        "pythagoras": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides. Written as a squared plus b squared equals c squared, this theorem is fundamental to geometry and has countless real-world applications.",
        "sine": "The sine function relates an angle to the ratio of the opposite side to the hypotenuse in a right triangle. It creates a wave pattern when graphed and is essential in describing periodic phenomena like sound and light waves.",
        "cosine": "The cosine function relates an angle to the ratio of the adjacent side to the hypotenuse in a right triangle. Like sine, it produces a wave pattern and is fundamental to understanding circular motion and oscillations.",
        "parabola": "A parabola is a U-shaped curve that represents a quadratic function. Every point on a parabola is equidistant from a fixed point called the focus and a fixed line called the directrix. Parabolas describe the path of projectiles and are used in satellite dishes.",
        "function": "A function is a mathematical relationship that assigns exactly one output to each input. Functions are the building blocks of mathematics and help us describe how one quantity depends on another.",
        "derivative": "The derivative measures the rate of change of a function at any point. It tells us how fast something is changing and is fundamental to calculus and physics for understanding motion and optimization.",
        "integral": "Integration is the process of finding the area under a curve. It's the reverse of differentiation and is used to calculate accumulated quantities like distance, area, and volume.",
    }
    
    for key, explanation in explanations.items():
        if key in concept_lower:
            return explanation
    
    return f"Let's explore the concept of {concept}. This is an important topic in mathematics that helps us understand patterns and relationships in the world around us. Watch carefully as we visualize this concept step by step."


def generate_animation_script(concept: str) -> dict:
    """Generate animation parameters based on the concept using OpenAI."""
    concept_lower = concept.lower().strip()
    
    if not OPENAI_API_KEY:
        return get_fallback_animation_script(concept_lower)
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": """You are an animation script generator for educational math videos using Manim.
Generate a JSON object with these fields:
- "type": one of ["circle", "triangle", "square", "rectangle", "pythagorean", "sine_wave", "cosine_wave", "parabola", "function_graph", "derivative", "integral", "custom_shape"]
- "title": a short title for the animation
- "color": primary color (BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, WHITE)
- "show_labels": boolean whether to show mathematical labels
- "show_formula": boolean whether to show the formula
Respond with only valid JSON."""
                },
                {
                    "role": "user",
                    "content": f"Generate animation script for: {concept}"
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=200
        )
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            return result
        return get_fallback_animation_script(concept_lower)
    except Exception as e:
        print(f"OpenAI animation script error: {e}")
        return get_fallback_animation_script(concept_lower)


def get_fallback_animation_script(concept: str) -> dict:
    """Fallback animation scripts when OpenAI is not available."""
    scripts = {
        "circle": {"type": "circle", "title": "The Circle", "color": "BLUE", "show_labels": True, "show_formula": True},
        "triangle": {"type": "triangle", "title": "The Triangle", "color": "GREEN", "show_labels": True, "show_formula": True},
        "square": {"type": "square", "title": "The Square", "color": "RED", "show_labels": True, "show_formula": True},
        "rectangle": {"type": "rectangle", "title": "The Rectangle", "color": "PURPLE", "show_labels": True, "show_formula": True},
        "pythagorean": {"type": "pythagorean", "title": "Pythagorean Theorem", "color": "YELLOW", "show_labels": True, "show_formula": True},
        "pythagoras": {"type": "pythagorean", "title": "Pythagorean Theorem", "color": "YELLOW", "show_labels": True, "show_formula": True},
        "sine": {"type": "sine_wave", "title": "Sine Function", "color": "BLUE", "show_labels": True, "show_formula": True},
        "cosine": {"type": "cosine_wave", "title": "Cosine Function", "color": "GREEN", "show_labels": True, "show_formula": True},
        "parabola": {"type": "parabola", "title": "The Parabola", "color": "ORANGE", "show_labels": True, "show_formula": True},
        "function": {"type": "function_graph", "title": "Function Graph", "color": "BLUE", "show_labels": True, "show_formula": True},
        "derivative": {"type": "derivative", "title": "The Derivative", "color": "RED", "show_labels": True, "show_formula": True},
        "integral": {"type": "integral", "title": "The Integral", "color": "PURPLE", "show_labels": True, "show_formula": True},
    }
    
    for key, script in scripts.items():
        if key in concept:
            return script
    
    return {"type": "circle", "title": concept.title(), "color": "BLUE", "show_labels": True, "show_formula": False}


def generate_manim_code(script: dict, output_name: str) -> str:
    """Generate Manim Python code based on the animation script."""
    anim_type = script.get("type", "circle")
    title = script.get("title", "Animation")
    color = script.get("color", "BLUE")
    show_labels = script.get("show_labels", True)
    show_formula = script.get("show_formula", True)
    
    code = f'''from manim import *
import numpy as np

class EducationalAnimation(Scene):
    def construct(self):
        # Title
        title = Text("{title}", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
'''
    
    if anim_type == "circle":
        code += f'''
        # Create circle
        circle = Circle(radius=2, color={color})
        center_dot = Dot(ORIGIN, color=WHITE)
        radius_line = Line(ORIGIN, circle.get_right(), color=YELLOW)
        
        self.play(Create(circle), run_time=1.5)
        self.play(Create(center_dot))
        self.play(Create(radius_line))
'''
        if show_labels:
            code += '''
        # Add labels
        radius_label = MathTex("r", font_size=36).next_to(radius_line, DOWN, buff=0.2)
        self.play(Write(radius_label))
'''
        if show_formula:
            code += '''
        # Show formulas
        area_formula = MathTex("A = \\\\pi r^2", font_size=40)
        circumference_formula = MathTex("C = 2\\\\pi r", font_size=40)
        formulas = VGroup(area_formula, circumference_formula).arrange(DOWN, buff=0.3)
        formulas.to_edge(DOWN)
        self.play(Write(area_formula))
        self.wait(0.5)
        self.play(Write(circumference_formula))
'''
    
    elif anim_type == "triangle":
        code += f'''
        # Create triangle
        triangle = Triangle(color={color}).scale(2)
        self.play(Create(triangle), run_time=1.5)
'''
        if show_labels:
            code += '''
        # Add vertex labels
        vertices = triangle.get_vertices()
        labels = VGroup(
            MathTex("A").next_to(vertices[0], UP),
            MathTex("B").next_to(vertices[1], DOWN + LEFT),
            MathTex("C").next_to(vertices[2], DOWN + RIGHT)
        )
        self.play(Write(labels))
'''
        if show_formula:
            code += '''
        # Show angle sum
        angle_sum = MathTex("\\\\angle A + \\\\angle B + \\\\angle C = 180^\\\\circ", font_size=36)
        angle_sum.to_edge(DOWN)
        self.play(Write(angle_sum))
'''
    
    elif anim_type == "square":
        code += f'''
        # Create square
        square = Square(side_length=3, color={color})
        self.play(Create(square), run_time=1.5)
'''
        if show_labels:
            code += '''
        # Add side label
        side_label = MathTex("s").next_to(square, DOWN, buff=0.3)
        self.play(Write(side_label))
'''
        if show_formula:
            code += '''
        # Show formulas
        area = MathTex("A = s^2", font_size=40)
        perimeter = MathTex("P = 4s", font_size=40)
        formulas = VGroup(area, perimeter).arrange(DOWN, buff=0.3)
        formulas.to_edge(DOWN)
        self.play(Write(formulas))
'''
    
    elif anim_type == "rectangle":
        code += f'''
        # Create rectangle
        rectangle = Rectangle(width=4, height=2.5, color={color})
        self.play(Create(rectangle), run_time=1.5)
'''
        if show_labels:
            code += '''
        # Add dimension labels
        width_label = MathTex("w").next_to(rectangle, DOWN, buff=0.3)
        height_label = MathTex("h").next_to(rectangle, RIGHT, buff=0.3)
        self.play(Write(width_label), Write(height_label))
'''
        if show_formula:
            code += '''
        # Show formulas
        area = MathTex("A = w \\\\times h", font_size=40)
        area.to_edge(DOWN)
        self.play(Write(area))
'''
    
    elif anim_type == "pythagorean":
        code += f'''
        # Create right triangle for Pythagorean theorem
        a, b = 2, 1.5
        c = (a**2 + b**2)**0.5
        
        triangle = Polygon(
            ORIGIN, RIGHT * a, RIGHT * a + UP * b,
            color={color}, fill_opacity=0.3
        )
        self.play(Create(triangle), run_time=1.5)
        
        # Right angle marker
        right_angle = RightAngle(
            Line(ORIGIN, RIGHT * a),
            Line(RIGHT * a, RIGHT * a + UP * b),
            length=0.3
        )
        self.play(Create(right_angle))
'''
        if show_labels:
            code += '''
        # Side labels
        a_label = MathTex("a").next_to(Line(ORIGIN, RIGHT * a), DOWN, buff=0.2)
        b_label = MathTex("b").next_to(Line(RIGHT * a, RIGHT * a + UP * b), RIGHT, buff=0.2)
        c_label = MathTex("c").next_to(Line(ORIGIN, RIGHT * a + UP * b), UP + LEFT, buff=0.2)
        self.play(Write(a_label), Write(b_label), Write(c_label))
'''
        if show_formula:
            code += '''
        # Pythagorean theorem formula
        theorem = MathTex("a^2 + b^2 = c^2", font_size=48)
        theorem.to_edge(DOWN)
        self.play(Write(theorem))
        
        # Highlight squares
        self.wait(0.5)
        box = SurroundingRectangle(theorem, color=YELLOW)
        self.play(Create(box))
'''
    
    elif anim_type == "sine_wave":
        code += f'''
        # Create axes
        axes = Axes(
            x_range=[-1, 7, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=8,
            y_length=4,
            axis_config={{"include_tip": True}}
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(axes_labels))
        
        # Draw sine wave
        sine_curve = axes.plot(lambda x: np.sin(x), color={color}, x_range=[0, 2*PI])
        self.play(Create(sine_curve), run_time=2)
'''
        if show_formula:
            code += '''
        # Show formula
        formula = MathTex("y = \\\\sin(x)", font_size=40)
        formula.to_edge(DOWN)
        self.play(Write(formula))
'''
    
    elif anim_type == "cosine_wave":
        code += f'''
        # Create axes
        axes = Axes(
            x_range=[-1, 7, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=8,
            y_length=4,
            axis_config={{"include_tip": True}}
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(axes_labels))
        
        # Draw cosine wave
        cosine_curve = axes.plot(lambda x: np.cos(x), color={color}, x_range=[0, 2*PI])
        self.play(Create(cosine_curve), run_time=2)
'''
        if show_formula:
            code += '''
        # Show formula
        formula = MathTex("y = \\\\cos(x)", font_size=40)
        formula.to_edge(DOWN)
        self.play(Write(formula))
'''
    
    elif anim_type == "parabola":
        code += f'''
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=5,
            axis_config={{"include_tip": True}}
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(axes_labels))
        
        # Draw parabola
        parabola = axes.plot(lambda x: x**2, color={color}, x_range=[-2.2, 2.2])
        self.play(Create(parabola), run_time=2)
        
        # Mark vertex
        vertex = Dot(axes.coords_to_point(0, 0), color=YELLOW)
        vertex_label = MathTex("(0, 0)").next_to(vertex, DOWN + RIGHT, buff=0.2)
        self.play(Create(vertex), Write(vertex_label))
'''
        if show_formula:
            code += '''
        # Show formula
        formula = MathTex("y = x^2", font_size=40)
        formula.to_edge(DOWN)
        self.play(Write(formula))
'''
    
    elif anim_type == "derivative":
        code += f'''
        # Create axes
        axes = Axes(
            x_range=[-2, 4, 1],
            y_range=[-2, 8, 2],
            x_length=7,
            y_length=5,
            axis_config={{"include_tip": True}}
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(axes_labels))
        
        # Original function
        curve = axes.plot(lambda x: x**2, color=BLUE, x_range=[-1.5, 2.8])
        curve_label = MathTex("f(x) = x^2", color=BLUE, font_size=32).to_edge(RIGHT).shift(UP)
        self.play(Create(curve), Write(curve_label))
        
        # Tangent line at a point
        x_val = 1.5
        tangent = axes.plot(lambda x: 2*x_val*(x - x_val) + x_val**2, color={color}, x_range=[0, 3])
        point = Dot(axes.coords_to_point(x_val, x_val**2), color=YELLOW)
        self.play(Create(point), Create(tangent))
        
        # Derivative
        derivative = axes.plot(lambda x: 2*x, color=GREEN, x_range=[-1, 2.5])
        deriv_label = MathTex("f'(x) = 2x", color=GREEN, font_size=32).next_to(curve_label, DOWN)
        self.play(Create(derivative), Write(deriv_label))
'''
    
    elif anim_type == "integral":
        code += f'''
        # Create axes
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=5,
            axis_config={{"include_tip": True}}
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(axes_labels))
        
        # Function curve
        curve = axes.plot(lambda x: 0.5*x**2, color={color}, x_range=[0, 3])
        self.play(Create(curve))
        
        # Area under curve
        area = axes.get_area(curve, x_range=[0.5, 2.5], color=BLUE, opacity=0.5)
        self.play(FadeIn(area))
        
        # Integral symbol
        integral = MathTex("\\\\int_a^b f(x) \\\\, dx", font_size=40)
        integral.to_edge(DOWN)
        self.play(Write(integral))
'''
    
    else:
        code += f'''
        # Default: create a circle
        shape = Circle(radius=2, color={color})
        self.play(Create(shape), run_time=1.5)
'''
    
    code += '''
        self.wait(2)
'''
    
    return code


def render_manim_animation(code: str, output_name: str) -> Path:
    """Render the Manim animation and return the path to the video file."""
    temp_dir = TEMP_DIR / output_name
    temp_dir.mkdir(exist_ok=True)
    
    script_path = temp_dir / "animation.py"
    with open(script_path, "w") as f:
        f.write(code)
    
    try:
        result = subprocess.run(
            [
                "python", "-m", "manim", "render",
                "-ql",
                "--format", "mp4",
                "--media_dir", str(temp_dir / "media"),
                str(script_path),
                "EducationalAnimation"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"Manim error: {result.stderr}")
            raise Exception(f"Manim rendering failed: {result.stderr}")
        
        video_path = temp_dir / "media" / "videos" / "animation" / "480p15" / "EducationalAnimation.mp4"
        
        if not video_path.exists():
            for root, dirs, files in os.walk(temp_dir / "media"):
                for file in files:
                    if file.endswith(".mp4"):
                        video_path = Path(root) / file
                        break
        
        if not video_path.exists():
            raise Exception("Video file not generated")
        
        return video_path
    
    except subprocess.TimeoutExpired:
        raise Exception("Animation rendering timed out")


def generate_tts_audio(text: str, output_path: Path) -> Path:
    """Generate text-to-speech audio using pyttsx3."""
    import pyttsx3
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    
    audio_file = output_path / "narration.wav"
    engine.save_to_file(text, str(audio_file))
    engine.runAndWait()
    
    if not audio_file.exists():
        raise Exception("TTS audio generation failed")
    
    return audio_file


def merge_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Merge video and audio using ffmpeg."""
    final_video = output_path
    
    video_info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    video_duration = float(video_info.stdout.strip()) if video_info.stdout.strip() else 10
    
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(final_video)
        ],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        shutil.copy(video_path, final_video)
    
    return final_video


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/api/generate")
async def generate_video(input_data: TextInput):
    """Main endpoint to generate educational animation video."""
    concept = input_data.text.strip()
    
    if not concept:
        raise HTTPException(status_code=400, detail="Please provide a concept to animate")
    
    video_id = str(uuid.uuid4())[:8]
    temp_dir = TEMP_DIR / video_id
    temp_dir.mkdir(exist_ok=True)
    
    try:
        explanation = generate_explanation(concept)
        
        animation_script = generate_animation_script(concept)
        
        manim_code = generate_manim_code(animation_script, video_id)
        
        video_path = render_manim_animation(manim_code, video_id)
        
        try:
            audio_path = generate_tts_audio(explanation, temp_dir)
            final_video_path = OUTPUT_DIR / f"{video_id}_final.mp4"
            merge_video_audio(video_path, audio_path, final_video_path)
        except Exception as tts_error:
            print(f"TTS/merge error: {tts_error}, using video without audio")
            final_video_path = OUTPUT_DIR / f"{video_id}_final.mp4"
            shutil.copy(video_path, final_video_path)
        
        return JSONResponse({
            "success": True,
            "video_url": f"/videos/{video_id}_final.mp4",
            "explanation": explanation,
            "animation_type": animation_script.get("type", "unknown"),
            "title": animation_script.get("title", concept)
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Educational Animation Generator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
