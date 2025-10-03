class TargetSegmentation:
    """
    Data structure representing a segmentation result for a target image.
    Stores the output file path and the resulting segmentation mask.

    Args:
        output_path (str): Path where the segmentation result should be saved.
        result_mask (np.ndarray): The segmentation mask (binary or multi-class) as a numpy array.
    """
    def __init__(self, output_path, result_mask):
        """
        Initialize the TargetSegmentation object.
        Args:
            output_path (str): Path where the segmentation result should be saved.
            result_mask (np.ndarray): The segmentation mask (binary or multi-class) as a numpy array.
        """
        self.output_path = output_path
        self.result_mask = result_mask