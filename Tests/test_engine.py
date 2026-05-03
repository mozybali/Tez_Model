from __future__ import annotations

import unittest
from unittest.mock import patch

from Model.engine import release_gpu_memory


class ReleaseGpuMemoryTest(unittest.TestCase):
    def test_cuda_cleanup_errors_are_suppressed(self) -> None:
        with (
            patch("Model.engine.gc.collect") as gc_collect,
            patch("Model.engine.torch.cuda.is_available", return_value=True),
            patch(
                "Model.engine.torch.cuda.empty_cache",
                side_effect=RuntimeError("CUDA out of memory"),
            ) as empty_cache,
            patch(
                "Model.engine.torch.cuda.ipc_collect",
                side_effect=RuntimeError("CUDA out of memory"),
            ) as ipc_collect,
        ):
            release_gpu_memory()

        gc_collect.assert_called_once()
        empty_cache.assert_called_once()
        ipc_collect.assert_called_once()

    def test_cleanup_runs_when_cuda_available(self) -> None:
        with (
            patch("Model.engine.gc.collect") as gc_collect,
            patch("Model.engine.torch.cuda.is_available", return_value=True),
            patch("Model.engine.torch.cuda.empty_cache") as empty_cache,
            patch("Model.engine.torch.cuda.ipc_collect") as ipc_collect,
        ):
            release_gpu_memory()

        gc_collect.assert_called_once()
        empty_cache.assert_called_once()
        ipc_collect.assert_called_once()

    def test_cleanup_skipped_when_cuda_unavailable(self) -> None:
        with (
            patch("Model.engine.gc.collect") as gc_collect,
            patch("Model.engine.torch.cuda.is_available", return_value=False),
            patch("Model.engine.torch.cuda.empty_cache") as empty_cache,
            patch("Model.engine.torch.cuda.ipc_collect") as ipc_collect,
        ):
            release_gpu_memory()

        gc_collect.assert_called_once()
        empty_cache.assert_not_called()
        ipc_collect.assert_not_called()

    def test_is_available_failure_short_circuits_cleanup(self) -> None:
        with (
            patch("Model.engine.gc.collect") as gc_collect,
            patch(
                "Model.engine.torch.cuda.is_available",
                side_effect=RuntimeError("driver error"),
            ),
            patch("Model.engine.torch.cuda.empty_cache") as empty_cache,
            patch("Model.engine.torch.cuda.ipc_collect") as ipc_collect,
        ):
            release_gpu_memory()

        gc_collect.assert_called_once()
        empty_cache.assert_not_called()
        ipc_collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
