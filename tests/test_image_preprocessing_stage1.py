from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from catalogbank_ocr.preprocessing.image_preprocessing import ImagePreprocessingConfig, apply_optional_preprocessing


class ImagePreprocessingStage1Tests(unittest.TestCase):
    def test_optional_preprocessing_enhances_and_denoises_image(self) -> None:
        image = Image.new("RGB", (100, 100), color="white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, 80, 80], outline="black", width=1)

        config = ImagePreprocessingConfig(dpi=200, denoise=True, enhance_resolution=True, resolution_scale=2.0)
        processed = apply_optional_preprocessing(image, config)

        self.assertGreaterEqual(processed.size[0], 200)
        self.assertGreaterEqual(processed.size[1], 200)


if __name__ == "__main__":
    unittest.main()
