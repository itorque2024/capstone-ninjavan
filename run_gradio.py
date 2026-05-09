import sys
from pathlib import Path

# Ensure the root directory is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the Gradio demo from the app module
from app.gradio_app import demo

if __name__ == "__main__":
    demo.launch()
