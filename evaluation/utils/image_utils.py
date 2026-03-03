"""
Image Processing Utilities

Common image processing functions for evaluation.
"""

import base64
import math
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image


def resize_image(
    image: Image.Image,
    max_pixels: Optional[int] = 512 * 512,
    min_pixels: Optional[int] = 338 * 338,
) -> Image.Image:
    """
    Resize image to fit within pixel bounds.
    
    Args:
        image: PIL Image object
        max_pixels: Maximum total pixels (width * height)
        min_pixels: Minimum total pixels
    
    Returns:
        Resized PIL Image
    """
    current_pixels = image.width * image.height
    
    if max_pixels and current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    
    if min_pixels and current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    
    return image


def encode_image(
    image_path: str,
    max_pixels: Optional[int] = 512 * 512,
    min_pixels: Optional[int] = 338 * 338,
    format: str = "JPEG",
) -> str:
    """
    Load, resize and encode image to base64.
    
    Args:
        image_path: Path to image file
        max_pixels: Maximum total pixels
        min_pixels: Minimum total pixels
        format: Output format (JPEG, PNG, etc.)
    
    Returns:
        Base64 encoded string
    """
    image = Image.open(image_path).convert("RGB")
    image.load()  # Prevent "too many open files" errors
    
    image = resize_image(image, max_pixels, min_pixels)
    
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_image_size(image_path: str) -> Tuple[int, int]:
    """Get image dimensions without loading full image."""
    with Image.open(image_path) as img:
        return img.size
