import gradio as gr
from PIL import Image
from models import LineartGenerator


class SketchToImageApp:
    """
    An application that combines a user’s annotated sketch with original lineart,
    and then generates an image using a ControlNet-based pipeline.
    """

    def __init__(self, lineart_generator: LineartGenerator):
        self.lineart_generator = lineart_generator

    def generate_image(self, brush_canvas: dict, uploaded_file) -> Image.Image:
        merged_lineart = self.lineart_generator.merge_lineart_and_brush(brush_canvas, uploaded_file)
        return self.lineart_generator.generate_image(merged_lineart, num_inference_steps=30)

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks() as app:
            gr.Markdown(
                "# Lineart & Color Mask With Controlnet\n"
                "Brush strokes will be applied behind the processed lineart so that the "
                "black lines always remain visible."
            )

            lineart_file_input = gr.File(
                label="Upload Lineart Sketch",
                file_types=["image"],
                file_count="single"
            )

            with gr.Row():
                brush_canvas_input = gr.Sketchpad(
                    label="Annotate Your Lineart",
                    type="numpy",
                    brush=gr.Brush(),
                )

            lineart_file_input.change(
                fn=self.lineart_generator.load_lineart_image,
                inputs=lineart_file_input,
                outputs=brush_canvas_input
            )

            generate_button = gr.Button("Generate")
            output_image = gr.Image(label="Generated Image")

            generate_button.click(
                fn=self.generate_image,
                inputs=[brush_canvas_input, lineart_file_input],
                outputs=output_image
            )

        return app
