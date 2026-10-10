"""Verify direct checkpoint inference against the independently shipped export."""

import unittest
from unittest.mock import patch

import torch

from checkpoint_policy import DEFAULT_CHECKPOINT, load_checkpoint_policy


class TestCheckpointPolicy(unittest.TestCase):
    def test_requested_cuda_does_not_silently_fall_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                load_checkpoint_policy(DEFAULT_CHECKPOINT, device="cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is unavailable")
    def test_cuda_actions_and_mujoco_controller(self):
        import mujoco
        import numpy as np
        from build_scene import TRACK_CONFIGS
        from controller import MoeStairController
        from teleop import KeyboardTeleop

        torch.set_num_threads(1)
        cpu_policy, _ = load_checkpoint_policy(DEFAULT_CHECKPOINT)
        gpu_policy, _ = load_checkpoint_policy(DEFAULT_CHECKPOINT, device="cuda:0")
        self.assertTrue(all(p.is_cuda for p in gpu_policy.parameters()))
        history = torch.randn(32, 225, generator=torch.Generator().manual_seed(17))
        with torch.inference_mode():
            torch.testing.assert_close(gpu_policy(history.cuda()).cpu(), cpu_policy(history),
                                       rtol=1e-4, atol=1e-5)

        scene = DEFAULT_CHECKPOINT.parents[2] / "svanm2_moe_stair_sim/scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        controller = MoeStairController(model, data, device="cuda:0")
        self.assertEqual(controller.device, torch.device("cuda:0"))
        teleop = KeyboardTeleop(controller, TRACK_CONFIGS)
        teleop.teleport_to_track(1)
        initial_x = float(data.qpos[0])
        for step in range(400):
            if step % 4 == 0:
                controller.policy_step((0.15, 0.0, 0.0))
            controller.physics_step()
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(float(data.qpos[0]), initial_x)
        self.assertGreater(float(data.qpos[2]), 0.24)

    def test_checkpoint_weights_and_actions_match_reference_export(self):
        torch.set_num_threads(1)
        # Loading the .pt must work without using TorchScript as a template.
        with patch("torch.jit.load", side_effect=AssertionError("Export used at runtime")):
            policy, update = load_checkpoint_policy(DEFAULT_CHECKPOINT)
        self.assertEqual(update, 90000)
        reference = torch.jit.load(str(DEFAULT_CHECKPOINT.with_name("offline_inference.ts")),
                                   map_location="cpu").eval()
        reference_weights = reference.state_dict()
        for key, tensor in policy.state_dict().items():
            export_key = key.replace("student_moe_encoder.", "encoder.", 1)
            torch.testing.assert_close(tensor, reference_weights[export_key], rtol=0, atol=0)

        generator = torch.Generator().manual_seed(42)
        with torch.inference_mode():
            for scale in (0.0, 0.1, 1.0, 5.0):
                history = torch.randn(32, 225, generator=generator) * scale
                actual = policy(history)
                self.assertEqual(actual.shape, (32, 12))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, reference(history), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
