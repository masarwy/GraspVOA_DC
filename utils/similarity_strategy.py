import numpy as np
import math
import cv2
from typing import Optional

from entities.real_camera import preprocess_depth_image


def normalize_depth_ignore_zeros(depth_im: np.ndarray) -> np.ndarray:
    """
    Normalize depth data based on its non-zero minimum and maximum values.

    Parameters:
    depth_im (numpy.ndarray): The depth image, where zero values are ignored in the normalization.

    Returns:
    numpy.ndarray: The normalized depth image, with non-zero values scaled to the 0-1 range.
    """

    non_zero_mask = depth_im > 0

    min_val = np.min(depth_im[non_zero_mask]) if np.any(non_zero_mask) else 0
    max_val = np.max(depth_im[non_zero_mask]) if np.any(non_zero_mask) else 1

    normalized_im = np.copy(depth_im)

    if max_val - min_val == 0:
        normalized_im[non_zero_mask] = 1
    else:
        normalized_im[non_zero_mask] = (depth_im[non_zero_mask] - min_val) / (max_val - min_val)

    return normalized_im


class SimilarityStrategy:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        raise NotImplementedError("This method should be implemented by subclasses.")


class StructureTermStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('Structure Term Similarity')

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        image_a = np.load(image_file_a)
        image_b = np.load(image_file_b)

        if a_is_real:
            image_a = preprocess_depth_image(image_a)
            image_a = normalize_depth_ignore_zeros(image_a / 1000.0)

        image_b = normalize_depth_ignore_zeros(image_b)

        exponent = 1
        return 10 * self.structural_term(image_a, image_b) ** exponent
        # return 100 * self.structural_term(image_a, image_b) ** exponent

    @staticmethod
    def structural_term(x: np.ndarray, y: np.ndarray, c3: float = 0.03 ** 2) -> float:
        """
        Compute the structural term of the SSIM index for two images.

        Parameters:
        x (numpy.ndarray): First image.
        y (numpy.ndarray): Second image.
        c3 (float): A small constant to stabilize division by small denominators.

        Returns:
        float: The structural term of the SSIM.
        """

        cov_xy = np.cov(x.flatten(), y.flatten())[0, 1]
        sigma_x = np.std(x)
        sigma_y = np.std(y)
        structural_similarity = (cov_xy + c3) / (sigma_x * sigma_y + c3)

        return structural_similarity


class ContourMatchStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('Contour Match Similarity')

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        ref_mask = np.load(image_file_a)
        ref_image = np.where(ref_mask != 0, 255, 0).astype(np.uint8)
        ref_contours, _ = cv2.findContours(ref_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ref_contour = max(ref_contours, key=cv2.contourArea)

        target_image = np.load(image_file_b)
        target_image = np.where(target_image != 0, 255, 0).astype(np.uint8)
        res = self.find_min_similarity(target_image, ref_contour)
        return 10 * math.exp(-10 * res)

    @staticmethod
    def find_min_similarity(target_image, ref_contour) -> float:
        # Find contours in the target image
        contours, _ = cv2.findContours(target_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize the minimum similarity score to a high value
        min_similarity = np.inf

        # Compare each contour in the target image to the reference contour
        for contour in contours:
            similarity = cv2.matchShapes(ref_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            min_similarity = min(min_similarity, similarity)

        return min_similarity


class IoUStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('IoU Similarity')

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        image_a = np.load(image_file_a)
        mask_a = np.where(image_a > 0, 1, 0)
        image_b = np.load(image_file_b)
        mask_b = np.where(image_b > 0, 1, 0)
        return self.compute_iou(mask_a, mask_b)

    @staticmethod
    def compute_iou(mask_a, mask_b):
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        iou = intersection / union if union != 0 else 0

        return iou


class HuMomentsStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('Hu Moments Similarity')

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        mask_a = np.load(image_file_a)
        mask_a = np.where(mask_a != 0, 255, 0).astype(np.uint8)
        mask_b = np.load(image_file_b)
        mask_b = np.where(mask_b != 0, 255, 0).astype(np.uint8)

        moments_a = cv2.HuMoments(cv2.moments(mask_a)).flatten()
        moments_b = cv2.HuMoments(cv2.moments(mask_b)).flatten()

        hu_score = np.sum(np.abs(moments_a - moments_b))

        return hu_score


class SimilarityContext:
    def __init__(self, strategy: Optional[SimilarityStrategy] = None) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: SimilarityStrategy):
        self._strategy = strategy

    def compare_images(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        return self._strategy(image_file_a, image_file_b, a_is_real)

    def get_strategy_name(self):
        return self._strategy.name
