import os
import re

import pandas as pd


class BmiPercentileCalculator:
    """
    Calculates the BMI percentile interval for a patient based on image and BMI table data.
    This class loads patient information and BMI reference percentiles, determines the patient's BMI percentile interval,
    and provides utility methods for extracting patient information from image paths and data tables.

    Args:
        image_info_path (str): Path to the CSV file containing patient and image information.
        bmi_table_path (str): Path to the CSV file containing BMI percentile reference values.
    """
    def __init__(self, image_info_path, bmi_table_path):
        """
        Initialize the BmiPercentileCalculator.
        Args:
            image_info_path (str): Path to the CSV file with patient/image info.
            bmi_table_path (str): Path to the CSV file with BMI percentiles.
        """
        self.image_info_path = image_info_path
        self.bmi_table_path = bmi_table_path
        
    def calculate_bmi_percentile_interval(self, image_path):
        """
        Calculate the BMI percentile interval for the patient associated with the given image.
        Args:
            image_path (str): Path to the image file (used to extract patient index).
        Returns:
            str: BMI percentile interval (e.g., '<P3', 'P3-P15', ..., '>P97').
        """
        all_data_info = pd.read_csv(self.image_info_path)
        bmi_table = pd.read_csv(self.bmi_table_path)
        patient_idx = self._parse_patient_index_from_image_path(image_path)
        bmi_percentile_interval = self._calculate_bmi_interval(all_data_info, patient_idx, bmi_table)
        return bmi_percentile_interval

    @staticmethod
    def _parse_patient_index_from_image_path(image_path):
        """
        Extract the patient index from an image file path using a regex pattern.
        Args:
            image_path (str): Path to the image file.
        Returns:
            str or None: Patient index if found, else None.
        """
        pattern = re.compile(r'^[^_]+_([^_]+(?:_\d+)+)(?=_\d{9,})')
        match = pattern.search(os.path.basename(image_path))
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _get_patient_info(all_data_info, patient_idx):
        """
        Retrieve sex, age, and BMI for a patient from the data table.
        Args:
            all_data_info (pd.DataFrame): DataFrame with patient/image info.
            patient_idx (str): Patient index to look up.
        Returns:
            tuple: (sex, age, bmi) for the patient.
        Raises:
            ValueError: If no data is found for the patient index.
        """
        patient_data = all_data_info[all_data_info['Patientenindex'] == patient_idx]
        if patient_data.empty:
            raise ValueError(f"No data found for patient index: {patient_idx}")

        sex = patient_data['Weiblich/Männlich'].iloc[0]
        age = patient_data['Alter'].iloc[0]
        bmi = patient_data['BMI'].iloc[0]
        return sex, age, bmi
    
    def _calculate_bmi_interval(self, all_data_info, patient_idx, bmi_table):
        """
        Determine the BMI percentile interval for a patient using reference percentiles.
        Args:
            all_data_info (pd.DataFrame): DataFrame with patient/image info.
            patient_idx (str): Patient index to look up.
            bmi_table (pd.DataFrame): DataFrame with BMI percentile reference values.
        Returns:
            str: BMI percentile interval (e.g., '<P3', 'P3-P15', ..., '>P97').
        Raises:
            ValueError: If no BMI data is found for the patient's age and sex.
        """
        sex, age, bmi = self._get_patient_info(all_data_info, patient_idx)
        sex_char = 'f' if sex == 'W' else 'm'

        bmi_row = bmi_table[(bmi_table['age_years'] == age) & (bmi_table['sex'] == sex_char)]

        if bmi_row.empty:
            raise ValueError(f"No BMI data found for age {age} and sex {sex_char}")

        percentiles = ['P3', 'P15', 'P50', 'P85', 'P97']
        bmi_values = bmi_row[percentiles].iloc[0]

        interval = ">P97"
        for i in range(len(bmi_values)):
            if bmi < bmi_values.iloc[i]:
                if i == 0:
                    interval = "<P3"
                else:
                    interval = f"{percentiles[i-1]}-{percentiles[i]}"
                break
        return interval