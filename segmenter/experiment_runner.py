"""
Module for running segmentation experiments.

This module provides the ExperimentRunner class which handles loading target images,
running the segmentation process via a provided segmenter, and recording timing
information for the experiment.
"""


import os
import time


class ExperimentRunner:
    """
    Class for running segmentation experiments and managing experiment results.
    Provides methods to execute experiments, collect results, and save experiment outputs.

    This class initializes with a segmenter and a directory of target images,
    executes the segmentation process, and saves duration metrics.
    """

    def __init__(self, segmenter, target_images_dir):
        """
        Initialize the ExperimentRunner.

        Args:
            segmenter: The segmenter instance to use for experiments. It must have
                methods `load_target_images(directory)` and `segment_images(images)`,
                and an attribute `output_dir`.
            target_images_dir: Path to the directory containing target images.
        """
        self.segmenter = segmenter
        self.target_images_dir = target_images_dir

    def run(self):
        """
        Run the segmentation experiment using the provided segmenter and configuration.

        Returns:
            dict: Results of the experiment, including total and average duration.
        """
        start_time = time.time()

        target_images = self.segmenter.load_target_images(self.target_images_dir)
        self.segmenter.segment_images(target_images)

        end_time = time.time()
        self.save_duration(start_time, end_time, len(target_images))

    def save_duration(self, start_time, end_time, num_images):
        """
        Calculate and save experiment duration.

        Args:
            start_time: Start time in seconds since the epoch.
            end_time: End time in seconds since the epoch.
            num_images: Number of images processed.

        The method computes total duration and average duration per image,
        formats them as HH:MM:SS.mmm strings, and writes them to a duration.txt
        file in the segmenter's output directory.
        """
        total_seconds = end_time - start_time

        # format total duration h:m:s.ms
        hrs = int(total_seconds // 3600)
        mins = int((total_seconds % 3600) // 60)
        secs_float = total_seconds % 60
        secs = int(secs_float)
        millis = int((secs_float - secs) * 1000)
        duration_str = f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

        # average per image
        if num_images > 0:
            avg_seconds = total_seconds / num_images
            avg_hrs = int(avg_seconds // 3600)
            avg_mins = int((avg_seconds % 3600) // 60)
            avg_secs_float = avg_seconds % 60
            avg_secs = int(avg_secs_float)
            avg_millis = int((avg_secs_float - avg_secs) * 1000)
            avg_str = f"{avg_hrs:02d}:{avg_mins:02d}:{avg_secs:02d}.{avg_millis:03d}"
        else:
            avg_str = "00:00:00"

        # write durations to file in output_dir
        duration_file = os.path.join(self.segmenter.output_dir, "duration.txt")
        with open(duration_file, "w") as f:
            f.write(f"Total duration: {duration_str}\n")
            f.write(f"Average per image: {avg_str}\n")
