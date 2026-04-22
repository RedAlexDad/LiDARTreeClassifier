import numpy as np


def augment_point_cloud(
    points: np.ndarray,
    noise_std: float = 0.005,
    num_points: int = 4096
) -> np.ndarray:
    idx = np.random.choice(points.shape[0], size=num_points, replace=True)
    augmented = points[idx].copy()
    augmented += np.random.normal(0, noise_std, augmented.shape)
    return augmented


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    aug_factor: int = 4,
    noise_std: float = 0.005,
    num_points: int = 4096
):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        for _ in range(aug_factor):
            X_aug.append(augment_point_cloud(X[i], noise_std, num_points))
            y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)


if __name__ == "__main__":
    X_dummy = np.random.randn(100, 4096, 3)
    y_dummy = np.random.randint(0, 3, size=100)

    X_aug, y_aug = augment_dataset(X_dummy, y_dummy, aug_factor=4, noise_std=0.005)
    print(f"Original: {X_dummy.shape}")
    print(f"Augmented: {X_aug.shape}")
    print(f"Y match: {len(y_aug) == len(X_aug)}")