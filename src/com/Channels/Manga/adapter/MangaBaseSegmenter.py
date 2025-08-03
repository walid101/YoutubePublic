"""
Create a base segmenter that class that must be implemented by external segmenters.
"""

from typing import List
import cv2
import numpy as np
from manga.src.models import ChapterDataModel, PageDataModel
from PIL import Image
from ozkapi.src.com.api.MangaApi import MangaApi
from abc import ABC, abstractmethod
import gc

import torch


class MangaBaseSegmenter(ABC):
    def __init__(self, api: MangaApi):
        self.api = api

    @abstractmethod
    def segment_pages(
        self, pages: List[PageDataModel], **kwargs
    ) -> List[List[Image.Image]]:
        """Segment pages into panels. Required"""
        pass

    """Helpers"""

    def convert_cv_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert an OpenCV image (NumPy array) to a PIL image."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        return Image.fromarray(image_rgb)

    def offload_model(self):
        model_offloaded = False
        if hasattr(self, "model") and self.model is not None:
            model_name = self.model.__class__.__name__
            try:
                original_model_ref = (
                    self.model
                )  # Keep a temporary ref for device check if needed below
                del self.model
                self.model = None
                print(
                    f"Model attribute '{model_name}' removed. Garbage collection will handle CPU memory."
                )
                model_offloaded = True

                # Check if the original model was on CUDA or if self.device indicates CUDA
                try:
                    # Attempt 1: Check model's parameters' device directly if it's a torch module
                    if isinstance(original_model_ref, torch.nn.Module):
                        if any(p.is_cuda for p in original_model_ref.parameters()):
                            print("Model was on CUDA, clearing cache...")
                            torch.cuda.empty_cache()
                            print("CUDA cache cleared.")
                    # Attempt 2: Fallback to checking self.device attribute
                    elif (
                        hasattr(self, "device")
                        and self.device is not None
                        and "cuda" in str(self.device)
                    ):
                        print("Device attribute indicates CUDA, clearing cache...")
                        torch.cuda.empty_cache()
                        print("CUDA cache cleared.")
                except Exception as e:
                    print(
                        f"Note: Could not definitively check model device or clear cache: {e}"
                    )

            except Exception as e:
                print(f"Error during model offload: {e}")
                # Ensure model attribute is None even if an error occurred during checks/cleanup
                if hasattr(self, "model"):
                    self.model = None

        else:
            print("No model found loaded in 'self.model' or already offloaded.")

        # Suggest garbage collection run (optional, Python does it automatically but can be hinted)
        if model_offloaded:
            print("Requesting garbage collection...")
            gc.collect()
        print("Offload process finished.")
