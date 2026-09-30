"""CPU-only coverage for the SALAD encoder's batch interface."""

import unittest

from PIL import Image
import torch

from countermine.mining.salad_encoder import SaladEncoder


class _MeanColorModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


class SaladEncoderStreamingTest(unittest.TestCase):
    def test_iter_encode_matches_encode_without_loading_weights(self) -> None:
        encoder = SaladEncoder.__new__(SaladEncoder)
        encoder.image_size = 322
        encoder.batch_size = 2
        encoder.device = torch.device("cpu")
        encoder.model = _MeanColorModel().eval()
        images = [Image.new("RGB", (1, 1), color=color) for color in (
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)
        )]

        batches = list(encoder.iter_encode(iter(images)))
        self.assertEqual([start for start, _ in batches], [0, 2, 4])
        self.assertEqual([tuple(batch.shape) for _, batch in batches], [(2, 3), (2, 3), (1, 3)])
        streamed = torch.cat([batch for _, batch in batches])
        torch.testing.assert_close(streamed, encoder.encode(images))
        torch.testing.assert_close(
            torch.linalg.vector_norm(streamed, dim=1), torch.ones(5), atol=1e-6, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
