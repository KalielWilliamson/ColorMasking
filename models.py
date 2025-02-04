import numpy as np
import torch
from PIL import Image
from controlnet_aux import LineartDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)


class LineartGenerator:

    def __init__(self, device: str = None, seed: int = 0):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.seed = seed
        self._initialize_models()

    def _initialize_models(self):
        self.lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
        self.controlnet = ControlNetModel.from_pretrained(checkpoint)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

    @staticmethod
    def load_lineart_image(uploaded_file, size: tuple = (512, 512)) -> np.ndarray:
        if not uploaded_file:
            return None

        with open(uploaded_file.name, "rb") as file_obj:
            image = Image.open(file_obj).convert("L")
            image = image.resize(size)
        return np.array(image, dtype=np.uint8)

    @staticmethod
    def merge_lineart_and_brush(brush_canvas, uploaded_file, size: tuple = (512, 512)) -> Image.Image:
        if brush_canvas is None or uploaded_file is None:
            return None

        # Reload and process the original lineart image
        with open(uploaded_file.name, "rb") as file_obj:
            lineart_image = Image.open(file_obj).convert("L")
            lineart_image = lineart_image.resize(size)

        lineart_rgba = lineart_image.convert("RGBA")
        processed_pixels = []
        for pixel in lineart_rgba.getdata():
            if pixel[0] > 240:
                processed_pixels.append((255, 255, 255, 0))
            else:
                processed_pixels.append((0, 0, 0, 255))
        lineart_rgba.putdata(processed_pixels)

        brush_layer = Image.fromarray(brush_canvas["composite"]).convert("RGBA")

        combined = Image.alpha_composite(brush_layer, lineart_rgba)
        return combined

    def generate_image(self, annotated_lineart: Image.Image,
                       num_inference_steps: int = 30) -> Image.Image:
        if annotated_lineart is None:
            raise ValueError("No annotated lineart provided!")

        annotated_lineart = annotated_lineart.resize((512, 512))
        refined_lineart = self.lineart_detector(annotated_lineart).convert("RGBA")
        generator = torch.manual_seed(self.seed)

        return self.pipe(
            prompt="A colorful image and high resolution image",
            image=refined_lineart,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt="monochrome, desaturated, low contrast"
        ).images[0]
