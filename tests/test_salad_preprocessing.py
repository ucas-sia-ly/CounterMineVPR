"""CPU-only check for SALAD's evaluation preprocessing convention."""

import unittest

from PIL import Image
import torch

from countermine.mining.salad_encoder import preprocess_image


class SaladPreprocessingTest(unittest.TestCase):
    def test_resize_and_normalize_rgb(self) -> None:
        image = Image.new("RGB", (1, 1), color=(255, 128, 0))
        result = preprocess_image(image)

        self.assertEqual(tuple(result.shape), (3, 322, 322))
        expected = torch.tensor(
            [
                (1.0 - 0.485) / 0.229,
                (128 / 255 - 0.456) / 0.224,
                (0.0 - 0.406) / 0.225,
            ]
        )
        torch.testing.assert_close(result[:, 0, 0], expected, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
