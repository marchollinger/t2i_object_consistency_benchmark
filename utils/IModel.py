from abc import ABC, abstractmethod

from PIL import Image


class Model(ABC):
    """Interface to support other models."""

    @abstractmethod
    def invoke(prompt: str) -> Image:
        """Invoke the model on a given prompt

        Args:
            prompt: The input prompt.

        Returns:
            The generated image.
        """
        ...
