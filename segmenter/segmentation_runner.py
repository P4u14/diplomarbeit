import os
import time


class SegmentationRunner:
    def __init__(self, segmenter, target_images_dir):
        self.segmenter = segmenter
        self.target_images_dir = target_images_dir

    def run(self):
        start_time = time.time()

        target_images = self.segmenter.load_target_images(self.target_images_dir)
        self.segmenter.segment_images(target_images)

        end_time = time.time()
        self.save_duration(start_time, end_time, len(target_images))

    def save_duration(self, start_time, end_time, num_images):
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