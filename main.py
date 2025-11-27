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
        "vector": "Vectors are mathematical objects that have both magnitude and direction. They are used to represent physical quantities like velocity, force, and displacement. Vector addition combines two vectors to produce a resultant vector.",
        "bubble sort": "Bubble sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted.",
        "sorting": "Sorting algorithms arrange elements in a specific order. Bubble sort repeatedly compares and swaps adjacent elements until the entire list is sorted from smallest to largest.",
        "graph": "Graphs visually represent mathematical functions by plotting points on a coordinate system. The x-axis represents input values and the y-axis shows corresponding outputs. Key features include maxima, minima, and intercepts.",
        "maxima": "In calculus, a maximum is the highest point on a curve within a given interval. Local maxima occur where the function changes from increasing to decreasing. These points are critical for optimization problems.",
        "minima": "In calculus, a minimum is the lowest point on a curve within a given interval. Local minima occur where the function changes from decreasing to increasing. Finding minima is essential in optimization.",
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
        "vector": {"type": "vector_addition", "title": "Vector Addition", "color": "BLUE", "show_labels": True, "show_formula": True},
        "arrow": {"type": "vector_addition", "title": "Vector Addition", "color": "BLUE", "show_labels": True, "show_formula": True},
        "bubble": {"type": "bubble_sort", "title": "Bubble Sort Algorithm", "color": "BLUE", "show_labels": True, "show_formula": False},
        "sort": {"type": "bubble_sort", "title": "Sorting Algorithm", "color": "GREEN", "show_labels": True, "show_formula": False},
        "maxima": {"type": "graph_extrema", "title": "Graph with Maxima and Minima", "color": "BLUE", "show_labels": True, "show_formula": True},
        "minima": {"type": "graph_extrema", "title": "Graph with Maxima and Minima", "color": "BLUE", "show_labels": True, "show_formula": True},
        "extrema": {"type": "graph_extrema", "title": "Finding Extrema", "color": "PURPLE", "show_labels": True, "show_formula": True},
    }
    
    for key, script in scripts.items():
        if key in concept:
            return script
    
    return {"type": "circle", "title": concept.title(), "color": "BLUE", "show_labels": True, "show_formula": False}


def generate_scene_breakdown(script: dict, concept: str) -> list:
    """Generate a scene-by-scene breakdown of the animation."""
    anim_type = script.get("type", "circle")
    title = script.get("title", "Animation")
    
    scene_templates = {
        "circle": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display title 'The Circle' at the top of the screen"},
            {"scene": 2, "name": "Circle Creation", "duration": "2s", "description": "Animate the circle being drawn from center outward"},
            {"scene": 3, "name": "Center Point", "duration": "1s", "description": "Show the center point of the circle"},
            {"scene": 4, "name": "Radius Line", "duration": "1s", "description": "Draw a line from center to edge showing the radius"},
            {"scene": 5, "name": "Labels", "duration": "1s", "description": "Add label 'r' to indicate the radius"},
            {"scene": 6, "name": "Formulas", "duration": "2s", "description": "Display Area and Circumference formulas"}
        ],
        "triangle": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display title 'The Triangle'"},
            {"scene": 2, "name": "Triangle Creation", "duration": "2s", "description": "Draw the three-sided polygon"},
            {"scene": 3, "name": "Vertex Labels", "duration": "1.5s", "description": "Label vertices A, B, and C"},
            {"scene": 4, "name": "Angle Sum", "duration": "2s", "description": "Show that angles sum to 180 degrees"}
        ],
        "pythagorean": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display 'Pythagorean Theorem'"},
            {"scene": 2, "name": "Right Triangle", "duration": "2s", "description": "Draw right triangle with sides a, b, c"},
            {"scene": 3, "name": "Right Angle Marker", "duration": "1s", "description": "Show the 90-degree angle marker"},
            {"scene": 4, "name": "Side Labels", "duration": "1.5s", "description": "Label sides a, b, and c (hypotenuse)"},
            {"scene": 5, "name": "Theorem Formula", "duration": "2s", "description": "Display a^2 + b^2 = c^2"},
            {"scene": 6, "name": "Highlight", "duration": "1s", "description": "Highlight the theorem with a box"}
        ],
        "vector_addition": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display 'Vector Addition'"},
            {"scene": 2, "name": "First Vector", "duration": "1.5s", "description": "Draw vector A as an arrow"},
            {"scene": 3, "name": "Second Vector", "duration": "1.5s", "description": "Draw vector B starting from A's tip"},
            {"scene": 4, "name": "Resultant Vector", "duration": "2s", "description": "Draw the resultant vector from origin to B's tip"},
            {"scene": 5, "name": "Labels", "duration": "1s", "description": "Label all vectors A, B, and A+B"},
            {"scene": 6, "name": "Formula", "duration": "1.5s", "description": "Show vector addition rule"}
        ],
        "bubble_sort": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display 'Bubble Sort Algorithm'"},
            {"scene": 2, "name": "Initial Array", "duration": "1.5s", "description": "Show unsorted array as colored bars"},
            {"scene": 3, "name": "Comparison Pass 1", "duration": "3s", "description": "Compare and swap adjacent elements"},
            {"scene": 4, "name": "Comparison Pass 2", "duration": "2.5s", "description": "Second pass through the array"},
            {"scene": 5, "name": "Final Sorted", "duration": "2s", "description": "Show the fully sorted array"},
            {"scene": 6, "name": "Complexity", "duration": "1.5s", "description": "Display time complexity O(n^2)"}
        ],
        "graph_extrema": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display 'Graph Maxima and Minima'"},
            {"scene": 2, "name": "Coordinate Axes", "duration": "1.5s", "description": "Draw x and y axes with labels"},
            {"scene": 3, "name": "Function Curve", "duration": "2.5s", "description": "Plot the function curve"},
            {"scene": 4, "name": "Maximum Point", "duration": "1.5s", "description": "Highlight the local maximum"},
            {"scene": 5, "name": "Minimum Point", "duration": "1.5s", "description": "Highlight the local minimum"},
            {"scene": 6, "name": "Labels", "duration": "1.5s", "description": "Add labels for extrema points"}
        ],
        "sine_wave": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display 'Sine Function'"},
            {"scene": 2, "name": "Coordinate Axes", "duration": "1.5s", "description": "Draw axes with x and y labels"},
            {"scene": 3, "name": "Wave Drawing", "duration": "3s", "description": "Animate the sine wave being traced"},
            {"scene": 4, "name": "Formula", "duration": "1.5s", "description": "Display y = sin(x)"}
        ],
        "derivative": [
            {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": "Display 'The Derivative'"},
            {"scene": 2, "name": "Coordinate Axes", "duration": "1.5s", "description": "Draw coordinate system"},
            {"scene": 3, "name": "Original Function", "duration": "2s", "description": "Plot f(x) = x^2"},
            {"scene": 4, "name": "Tangent Line", "duration": "2s", "description": "Show tangent line at a point"},
            {"scene": 5, "name": "Derivative Curve", "duration": "2s", "description": "Plot the derivative f'(x) = 2x"}
        ],
    }
    
    if anim_type in scene_templates:
        return scene_templates[anim_type]
    
    return [
        {"scene": 1, "name": "Title Introduction", "duration": "1.5s", "description": f"Display title '{title}'"},
        {"scene": 2, "name": "Main Animation", "duration": "3s", "description": f"Animate the concept of {concept}"},
        {"scene": 3, "name": "Labels and Details", "duration": "2s", "description": "Add labels and annotations"},
        {"scene": 4, "name": "Conclusion", "duration": "2s", "description": "Display summary or formula"}
    ]


def generate_manim_code(script: dict, output_name: str) -> str:
    """Generate Manim Python code based on the animation script (LaTeX-free version)."""
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
        radius_label = Text("r", font_size=28, color=YELLOW).next_to(radius_line, DOWN, buff=0.2)
        self.play(Write(radius_label))
'''
        if show_formula:
            code += '''
        # Show formulas
        area_formula = Text("Area = pi * r^2", font_size=32)
        circumference_formula = Text("Circumference = 2 * pi * r", font_size=32)
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
            Text("A", font_size=28).next_to(vertices[0], UP),
            Text("B", font_size=28).next_to(vertices[1], DOWN + LEFT),
            Text("C", font_size=28).next_to(vertices[2], DOWN + RIGHT)
        )
        self.play(Write(labels))
'''
        if show_formula:
            code += '''
        # Show angle sum
        angle_sum = Text("Angle A + Angle B + Angle C = 180 degrees", font_size=28)
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
        side_label = Text("s", font_size=28).next_to(square, DOWN, buff=0.3)
        self.play(Write(side_label))
'''
        if show_formula:
            code += '''
        # Show formulas
        area = Text("Area = s^2", font_size=32)
        perimeter = Text("Perimeter = 4s", font_size=32)
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
        width_label = Text("w", font_size=28).next_to(rectangle, DOWN, buff=0.3)
        height_label = Text("h", font_size=28).next_to(rectangle, RIGHT, buff=0.3)
        self.play(Write(width_label), Write(height_label))
'''
        if show_formula:
            code += '''
        # Show formulas
        area = Text("Area = w x h", font_size=32)
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
        a_label = Text("a", font_size=28).next_to(Line(ORIGIN, RIGHT * a), DOWN, buff=0.2)
        b_label = Text("b", font_size=28).next_to(Line(RIGHT * a, RIGHT * a + UP * b), RIGHT, buff=0.2)
        c_label = Text("c", font_size=28).next_to(Line(ORIGIN, RIGHT * a + UP * b), UP + LEFT, buff=0.2)
        self.play(Write(a_label), Write(b_label), Write(c_label))
'''
        if show_formula:
            code += '''
        # Pythagorean theorem formula
        theorem = Text("a^2 + b^2 = c^2", font_size=40)
        theorem.to_edge(DOWN)
        self.play(Write(theorem))
        
        # Highlight
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
        x_label = Text("x", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, UP)
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Draw sine wave
        sine_curve = axes.plot(lambda x: np.sin(x), color={color}, x_range=[0, 2*PI])
        self.play(Create(sine_curve), run_time=2)
'''
        if show_formula:
            code += '''
        # Show formula
        formula = Text("y = sin(x)", font_size=32)
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
        x_label = Text("x", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, UP)
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Draw cosine wave
        cosine_curve = axes.plot(lambda x: np.cos(x), color={color}, x_range=[0, 2*PI])
        self.play(Create(cosine_curve), run_time=2)
'''
        if show_formula:
            code += '''
        # Show formula
        formula = Text("y = cos(x)", font_size=32)
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
        x_label = Text("x", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, UP)
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Draw parabola
        parabola = axes.plot(lambda x: x**2, color={color}, x_range=[-2.2, 2.2])
        self.play(Create(parabola), run_time=2)
        
        # Mark vertex
        vertex = Dot(axes.coords_to_point(0, 0), color=YELLOW)
        vertex_label = Text("(0, 0)", font_size=24).next_to(vertex, DOWN + RIGHT, buff=0.2)
        self.play(Create(vertex), Write(vertex_label))
'''
        if show_formula:
            code += '''
        # Show formula
        formula = Text("y = x^2", font_size=32)
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
        x_label = Text("x", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, UP)
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Original function
        curve = axes.plot(lambda x: x**2, color=BLUE, x_range=[-1.5, 2.8])
        curve_label = Text("f(x) = x^2", color=BLUE, font_size=24).to_edge(RIGHT).shift(UP)
        self.play(Create(curve), Write(curve_label))
        
        # Tangent line at a point
        x_val = 1.5
        tangent = axes.plot(lambda x: 2*x_val*(x - x_val) + x_val**2, color={color}, x_range=[0, 3])
        point = Dot(axes.coords_to_point(x_val, x_val**2), color=YELLOW)
        self.play(Create(point), Create(tangent))
        
        # Derivative
        derivative = axes.plot(lambda x: 2*x, color=GREEN, x_range=[-1, 2.5])
        deriv_label = Text("f'(x) = 2x", color=GREEN, font_size=24).next_to(curve_label, DOWN)
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
        x_label = Text("x", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, UP)
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Function curve
        curve = axes.plot(lambda x: 0.5*x**2, color={color}, x_range=[0, 3])
        self.play(Create(curve))
        
        # Area under curve
        area = axes.get_area(curve, x_range=[0.5, 2.5], color=BLUE, opacity=0.5)
        self.play(FadeIn(area))
        
        # Integral description
        integral = Text("Integral: Area under the curve", font_size=32)
        integral.to_edge(DOWN)
        self.play(Write(integral))
'''
    
    elif anim_type == "vector_addition":
        code += f'''
        # Vector Addition Animation
        # First vector (A)
        vector_a = Arrow(ORIGIN, RIGHT * 2 + UP * 1, color=BLUE, buff=0)
        label_a = Text("A", font_size=24, color=BLUE).next_to(vector_a, UP, buff=0.1)
        self.play(GrowArrow(vector_a), Write(label_a))
        
        # Second vector (B) - starts from tip of A
        start_b = vector_a.get_end()
        vector_b = Arrow(start_b, start_b + RIGHT * 1 + UP * 2, color=GREEN, buff=0)
        label_b = Text("B", font_size=24, color=GREEN).next_to(vector_b, RIGHT, buff=0.1)
        self.play(GrowArrow(vector_b), Write(label_b))
        
        # Resultant vector (A + B)
        resultant = Arrow(ORIGIN, vector_b.get_end(), color={color}, buff=0)
        label_r = Text("A + B", font_size=24, color={color}).next_to(resultant, LEFT, buff=0.1)
        self.play(GrowArrow(resultant), Write(label_r))
        
        # Show dashed lines for parallelogram
        dashed_a = DashedLine(vector_b.get_end(), vector_b.get_end() - (RIGHT * 2 + UP * 1), color=BLUE)
        dashed_b = DashedLine(vector_a.get_end(), vector_a.get_end() + (RIGHT * 1 + UP * 2), color=GREEN)
        self.play(Create(dashed_a), Create(dashed_b))
'''
        if show_formula:
            code += '''
        # Formula
        formula = Text("Vector Sum: A + B = Resultant", font_size=28)
        formula.to_edge(DOWN)
        self.play(Write(formula))
'''
    
    elif anim_type == "bubble_sort":
        code += f'''
        # Bubble Sort Animation
        # Create bars representing array values
        values = [5, 2, 8, 1, 9, 4]
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        bars = VGroup()
        
        for i, val in enumerate(values):
            bar = Rectangle(width=0.8, height=val * 0.4, fill_opacity=0.8)
            bar.set_fill(colors[i])
            bar.set_stroke(WHITE, 2)
            bar.move_to(LEFT * 3 + RIGHT * i * 1.2 + UP * (val * 0.2 - 1))
            bars.add(bar)
        
        self.play(Create(bars))
        self.wait(0.5)
        
        # Bubble sort passes (simplified animation)
        # Pass 1: Compare and swap
        for i in range(len(values) - 1):
            # Highlight comparison
            highlight = SurroundingRectangle(VGroup(bars[i], bars[i+1]), color=WHITE)
            self.play(Create(highlight), run_time=0.3)
            
            if values[i] > values[i+1]:
                # Swap
                values[i], values[i+1] = values[i+1], values[i]
                self.play(
                    bars[i].animate.shift(RIGHT * 1.2),
                    bars[i+1].animate.shift(LEFT * 1.2),
                    run_time=0.4
                )
                bars[i], bars[i+1] = bars[i+1], bars[i]
            
            self.play(FadeOut(highlight), run_time=0.2)
        
        # Show sorted label
        sorted_label = Text("Sorted!", font_size=36, color=GREEN)
        sorted_label.to_edge(DOWN)
        self.play(Write(sorted_label))
'''
    
    elif anim_type == "graph_extrema":
        code += f'''
        # Graph with Maxima and Minima
        axes = Axes(
            x_range=[-1, 7, 1],
            y_range=[-2, 4, 1],
            x_length=8,
            y_length=5,
            axis_config={{"include_tip": True}}
        )
        x_label = Text("x", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, UP)
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Plot a function with local max and min: y = sin(x) * 2
        curve = axes.plot(
            lambda x: 2 * np.sin(x),
            color={color},
            x_range=[0, 2*PI]
        )
        self.play(Create(curve), run_time=2)
        
        # Mark maximum at x = pi/2
        max_x = PI / 2
        max_point = Dot(axes.coords_to_point(max_x, 2), color=RED, radius=0.15)
        max_label = Text("Maximum", font_size=20, color=RED).next_to(max_point, UP)
        self.play(Create(max_point), Write(max_label))
        
        # Mark minimum at x = 3*pi/2
        min_x = 3 * PI / 2
        min_point = Dot(axes.coords_to_point(min_x, -2), color=GREEN, radius=0.15)
        min_label = Text("Minimum", font_size=20, color=GREEN).next_to(min_point, DOWN)
        self.play(Create(min_point), Write(min_label))
'''
        if show_formula:
            code += '''
        # Formula
        formula = Text("Finding peaks and valleys in functions", font_size=24)
        formula.to_edge(DOWN)
        self.play(Write(formula))
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
    """Generate text-to-speech audio using espeak directly via subprocess."""
    output_path.mkdir(parents=True, exist_ok=True)
    audio_file = output_path / "narration.wav"
    
    text_clean = text.replace('"', '\\"').replace("'", "\\'")
    
    result = subprocess.run(
        [
            "espeak",
            "-w", str(audio_file),
            "-s", "150",
            "-a", "180",
            text_clean
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"espeak error: {result.stderr}")
        raise Exception(f"TTS audio generation failed: {result.stderr}")
    
    if not audio_file.exists():
        raise Exception("TTS audio generation failed - file not created")
    
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
        raise HTTPException(status_code=400, detail="Please provide a concept to animate. Try something like 'circle', 'Pythagorean theorem', or 'bubble sort'.")
    
    video_id = str(uuid.uuid4())[:8]
    temp_dir = TEMP_DIR / video_id
    temp_dir.mkdir(exist_ok=True)
    
    try:
        explanation = generate_explanation(concept)
        
        animation_script = generate_animation_script(concept)
        
        scene_breakdown = generate_scene_breakdown(animation_script, concept)
        
        manim_code = generate_manim_code(animation_script, video_id)
        
        video_path = render_manim_animation(manim_code, video_id)
        
        has_audio = False
        try:
            audio_path = generate_tts_audio(explanation, temp_dir)
            final_video_path = OUTPUT_DIR / f"{video_id}_final.mp4"
            merge_video_audio(video_path, audio_path, final_video_path)
            has_audio = True
        except Exception as tts_error:
            print(f"TTS/merge error: {tts_error}, using video without audio")
            final_video_path = OUTPUT_DIR / f"{video_id}_final.mp4"
            shutil.copy(video_path, final_video_path)
        
        return JSONResponse({
            "success": True,
            "video_url": f"/videos/{video_id}_final.mp4",
            "explanation": explanation,
            "animation_type": animation_script.get("type", "unknown"),
            "title": animation_script.get("title", concept),
            "scene_breakdown": scene_breakdown,
            "animation_script": animation_script,
            "has_audio": has_audio,
            "generated_code_preview": manim_code[:500] + "..." if len(manim_code) > 500 else manim_code
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        
        if "rendering failed" in error_msg.lower():
            user_friendly_error = f"The animation rendering encountered an issue. This might be due to a complex concept. Please try a simpler term like 'circle' or 'triangle'. Technical details: {error_msg[:200]}"
        elif "timeout" in error_msg.lower():
            user_friendly_error = "The animation took too long to generate. Please try a simpler concept."
        elif "not found" in error_msg.lower():
            user_friendly_error = "A required component is missing. Please try again or use a different concept."
        else:
            user_friendly_error = f"Something went wrong while generating your animation. Please try again with a different concept. Error: {error_msg[:150]}"
        
        raise HTTPException(status_code=500, detail=user_friendly_error)
    
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
