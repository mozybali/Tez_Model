from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_functional


class Compose3D:
    def __init__(self, transforms: list[object] | tuple[object, ...]) -> None:
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            volume = transform(volume)
        return volume


class RandomFlip3D:
    def __init__(self, axes: tuple[int, ...] = (0, 1, 2), probability: float = 0.5) -> None:
        self.axes = axes
        self.probability = probability

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        augmented = volume
        for axis in self.axes:
            if torch.rand(1).item() < self.probability:
                augmented = torch.flip(augmented, dims=(axis + 1,))
        return augmented


class RandomAffine3D:
    def __init__(
        self,
        probability: float = 0.6,
        rotation_degrees: float = 10.0,
        translation_fraction: float = 0.05,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
    ) -> None:
        self.probability = probability
        self.rotation_degrees = rotation_degrees
        self.translation_fraction = translation_fraction
        self.scale_min = scale_min
        self.scale_max = scale_max

    @staticmethod
    def _rotation_matrix(angles_rad: torch.Tensor) -> torch.Tensor:
        angle_x, angle_y, angle_z = angles_rad
        sin_x, cos_x = torch.sin(angle_x), torch.cos(angle_x)
        sin_y, cos_y = torch.sin(angle_y), torch.cos(angle_y)
        sin_z, cos_z = torch.sin(angle_z), torch.cos(angle_z)

        zero = torch.zeros(1, dtype=torch.float32, device=angles_rad.device).squeeze()
        one = torch.ones(1, dtype=torch.float32, device=angles_rad.device).squeeze()

        rotation_x = torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos_x, -sin_x]),
            torch.stack([zero, sin_x, cos_x]),
        ])
        rotation_y = torch.stack([
            torch.stack([cos_y, zero, sin_y]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin_y, zero, cos_y]),
        ])
        rotation_z = torch.stack([
            torch.stack([cos_z, -sin_z, zero]),
            torch.stack([sin_z, cos_z, zero]),
            torch.stack([zero, zero, one]),
        ])
        return rotation_z @ rotation_y @ rotation_x

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.probability:
            return volume

        device = volume.device
        degrees = torch.empty(3, device=device).uniform_(-self.rotation_degrees, self.rotation_degrees)
        angles = degrees * math.pi / 180.0
        scale = torch.empty(1, device=device).uniform_(self.scale_min, self.scale_max).item()
        translation = torch.empty(3, device=device).uniform_(
            -self.translation_fraction,
            self.translation_fraction,
        )

        theta = torch.zeros((1, 3, 4), dtype=torch.float32, device=device)
        theta[0, :, :3] = self._rotation_matrix(angles) * scale
        theta[0, :, 3] = translation

        batched = volume.unsqueeze(0)
        grid = torch_functional.affine_grid(theta, batched.size(), align_corners=False)
        warped = torch_functional.grid_sample(
            batched,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped.squeeze(0)


class RandomMorphology3D:
    def __init__(self, probability: float = 0.1) -> None:
        self.probability = probability

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.probability:
            return volume

        batched = volume.unsqueeze(0)
        if torch.rand(1).item() < 0.5:
            morphed = torch_functional.max_pool3d(batched, kernel_size=3, stride=1, padding=1)
        else:
            morphed = 1.0 - torch_functional.max_pool3d(1.0 - batched, kernel_size=3, stride=1, padding=1)
        return morphed.squeeze(0)


def build_train_augmentations(config) -> Compose3D | None:
    if not config.enabled:
        return None
    return Compose3D(
        [
            RandomFlip3D(axes=config.flip_axes, probability=config.flip_probability),
            RandomAffine3D(
                probability=config.affine_probability,
                rotation_degrees=config.rotation_degrees,
                translation_fraction=config.translation_fraction,
                scale_min=config.scale_min,
                scale_max=config.scale_max,
            ),
            RandomMorphology3D(probability=config.morphology_probability),
        ]
    )

