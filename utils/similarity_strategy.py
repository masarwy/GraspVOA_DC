import numpy as np
import math
import cv2
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

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
        if a_is_real:
            ref_mask = preprocess_depth_image(ref_mask)

        ref_image = np.where(ref_mask != 0, 255, 0).astype(np.uint8)
        ref_contours, _ = cv2.findContours(ref_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ref_contour = max(ref_contours, key=cv2.contourArea)

        target_image = np.load(image_file_b)
        target_image = np.where(target_image != 0, 255, 0).astype(np.uint8)
        res, contour_2 = self.find_min_similarity(target_image, ref_contour, visualize=False)
        res2 = self.calculate_centroid_distance(contour1=ref_contour, contour2=contour_2)

        return 1 / (1 + res + res2)

    @staticmethod
    def calculate_centroid_distance(contour1, contour2):
        # Calculate the moments of the contours
        M1 = cv2.moments(contour1)
        M2 = cv2.moments(contour2)

        # Calculate the centroids
        if M1["m00"] != 0 and M2["m00"] != 0:
            cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
            cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
        else:
            # Handling the case of division by zero
            cx1, cy1, cx2, cy2 = 0, 0, 0, 0

        # Calculate the Euclidean distance between the centroids
        distance = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        return distance

    @staticmethod
    def find_min_similarity(target_image, ref_contour, visualize=False) -> Tuple[float, List]:
        contours, _ = cv2.findContours(target_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_similarity = np.inf
        min_similarity_contour = None

        vis_image = None
        if visualize:
            # Create a white canvas for visualization if visualization is enabled
            h, w = target_image.shape[:2]
            vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255

        for contour in contours:
            similarity = cv2.matchShapes(ref_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if similarity < min_similarity:
                min_similarity = similarity
                min_similarity_contour = contour

            if visualize:
                # Draw the current contour and the reference contour on the canvas for comparison
                cv2.drawContours(vis_image, [contour], -1, (255, 0, 0), 2)  # Draw current contour in blue
                cv2.drawContours(vis_image, [ref_contour], -1, (0, 255, 0), 2)  # Draw reference contour in green

                # Show the visualization
                plt.figure(figsize=(5, 5))
                plt.title(f"Contour Similarity: {similarity:.4f}")
                plt.imshow(vis_image)
                plt.axis('off')
                plt.show()

        if visualize and min_similarity_contour is not None:
            # Highlight the most similar contour in a different color if visualization is enabled
            cv2.drawContours(vis_image, [min_similarity_contour], -1, (0, 0, 255),
                             3)  # Draw most similar contour in red
            plt.figure(figsize=(5, 5))
            plt.title("Most Similar Contour")
            plt.imshow(vis_image)
            plt.axis('off')
            plt.show()

        return min_similarity, min_similarity_contour


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


class TemplateMatchingStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('Template Matching Similarity')

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        # Load .npy images
        img_a = np.load(image_file_a)
        img_b = np.load(image_file_b)

        if a_is_real:
            img_a = preprocess_depth_image(img_a)

        # Ensure images are 2D (grayscale)
        if img_a.ndim > 2:
            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        if img_b.ndim > 2:
            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        # Ensure images are of type float32 or convert them to uint8
        if img_a.dtype != np.uint8:
            img_a = (img_a * 255).astype(np.uint8)
        if img_b.dtype != np.uint8:
            img_b = (img_b * 255).astype(np.uint8)

        # Apply template matching
        res = cv2.matchTemplate(img_a, img_b, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        # Return the highest matching score
        return max_val


class FeatureBasedMatchingStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('Feature Based Matching Similarity')

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        # Load .npy images
        img_a = np.load(image_file_a)
        img_b = np.load(image_file_b)

        if a_is_real:
            img_a = preprocess_depth_image(img_a)

        # Ensure images are in BGR if they are grayscale for feature detection
        if img_a.ndim == 2:
            img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
        if img_b.ndim == 2:
            img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and descriptors
        kp_a, des_a = orb.detectAndCompute(img_a, None)
        kp_b, des_b = orb.detectAndCompute(img_b, None)

        # Create BFMatcher object and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_a, des_b)

        # Calculate the similarity as the inverse of the average distance of the best matches
        if matches:
            average_distance = sum(match.distance for match in matches) / len(matches)
            similarity = 1 / (1 + average_distance)  # Transform distance to similarity
        else:
            similarity = 0  # No matches found

        return similarity


class EMDSimilarityStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('Earth Mover\'s Distance Similarity')

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        img_a = np.load(image_file_a)

        if a_is_real:
            img_a = preprocess_depth_image(img_a)
        img_a = img_a.flatten()

        img_b = np.load(image_file_b).flatten()

        # Normalize the histograms to sum to 1, if not already
        if np.sum(img_a) != 1.0:
            img_a = img_a / np.sum(img_a)
        if np.sum(img_b) != 1.0:
            img_b = img_b / np.sum(img_b)

        # Calculate EMD
        emd_value = wasserstein_distance(img_a, img_b)

        # Invert the distance to represent similarity (lower distance = higher similarity)
        similarity = 1 / (1 + emd_value)

        return similarity


class ModEMDSimilarityStrategy(SimilarityStrategy):
    def __init__(self):
        super().__init__('Earth Mover\'s Distance Similarity')

    @staticmethod
    def create_image_signature(image, n_clusters=5):
        # Assuming image is 2D (grayscale), flatten it and get pixel coordinates
        x, y = np.indices(image.shape)
        features = np.stack((image.flatten(), x.flatten(), y.flatten()), axis=1)

        # Use KMeans to cluster features
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)

        # Cluster centers will be the signature
        signature = kmeans.cluster_centers_

        # Ensure non-negative weights
        signature[:, 0] = np.abs(signature[:, 0])  # Use absolute values to avoid negative weights

        # Normalize the weights
        signature[:, 0] /= signature[:, 0].sum()

        return signature

    @staticmethod
    def calculate_emd_signature(signature_a, signature_b):
        # Extract weights and features from signatures
        weights_a, features_a = signature_a[:, 0], signature_a[:, 1:]
        weights_b, features_b = signature_b[:, 0], signature_b[:, 1:]

        # Compute EMD using weights and features
        emd_total = sum(
            wasserstein_distance(features_a[:, i], features_b[:, i], u_weights=weights_a, v_weights=weights_b) for i in
            range(features_a.shape[1]))

        return emd_total

    def __call__(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        # Load and optionally preprocess the images
        img_a = np.load(image_file_a)
        if a_is_real:
            img_a = preprocess_depth_image(img_a)  # Ensure this function exists and is correct

        img_b = np.load(image_file_b)

        # Create signatures from images
        signature_a = self.create_image_signature(img_a)
        signature_b = self.create_image_signature(img_b)

        # Calculate EMD using signatures
        emd_value = self.calculate_emd_signature(signature_a, signature_b)

        # Invert the distance to represent similarity (lower distance = higher similarity)
        similarity = 1000 / (1 + emd_value)

        return similarity


class SimilarityContext:
    def __init__(self, strategy: Optional[SimilarityStrategy] = None) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: SimilarityStrategy):
        self._strategy = strategy

    def compare_images(self, image_file_a: str, image_file_b: str, a_is_real: bool = False) -> float:
        return self._strategy(image_file_a, image_file_b, a_is_real)

    def get_strategy_name(self):
        return self._strategy.name
