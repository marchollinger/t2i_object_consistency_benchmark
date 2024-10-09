from abc import ABC, abstractmethod
import io
from pathlib import Path

from PIL import Image
from transformers import AlignModel, AlignProcessor, CLIPModel, CLIPProcessor


class SimScore(ABC):
    """Subclasses of SimScore must provide 'self.processor' and 'self.model'."""

    def calculate_score(self, image: Path | io.BytesIO, prompt: str) -> float:
        image = Image.open(image)
        input = self.processor(
            text=prompt, images=image, return_tensors="pt", padding=True
        ).to(self.model.device)
        output = self.model(**input)
        logits_per_image = output.logits_per_image
        return logits_per_image.item()


class CLIPScore(SimScore):
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")


class ALIGNScore:
    def __init__(self):
        self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base")
