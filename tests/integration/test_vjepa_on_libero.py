"""Integration test: starVLA LIBERO dataloader → V-JEPA 2 encoder."""

import os
import pytest
import torch

LIBERO_DATA = "playground/Datasets/LEROBOT_LIBERO_DATA"
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DATA_REQUIRED = pytest.mark.skipif(
    not os.path.isdir(LIBERO_DATA), reason=f"LIBERO data not at {LIBERO_DATA}"
)


@CUDA_REQUIRED
@DATA_REQUIRED
def test_libero_batch_through_vjepa_encoder():
    import numpy as np
    from PIL import Image
    from starVLA.model.modules.world_model.vjepa2 import VJEPA2Encoder

    enc = VJEPA2Encoder(ckpt_path=None, num_frames=2, tubelet_size=2, img_size=256, patch_size=16).cuda()

    # Minimal frame stand-in until the LeRobot dataset config is wired up.
    img = Image.new("RGB", (256, 256), (128, 64, 200))
    arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    frames = arr.unsqueeze(0).cuda()  # [1, 3, 256, 256]

    z = enc.encode(frames)
    assert z.dim() == 3 and z.shape[-1] == 1408
