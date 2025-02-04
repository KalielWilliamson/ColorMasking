from models import LineartGenerator
from ui import SketchToImageApp


if __name__ == "__main__":
    lineart_generator = LineartGenerator()
    app = SketchToImageApp(lineart_generator).create_interface()
    app.launch()

