import os
import sys
from pathlib import Path

import pandas as pd
import pytest

project_root_content = "src"
current_folder = os.path.dirname(os.path.realpath(__name__))
while project_root_content not in os.listdir(current_folder):
    current_folder = str(Path(os.path.join(current_folder, "../")).resolve())
sys.path.insert(0, current_folder)

import src.frontend.helpers as fe_helpers
from src.algo import GroupDistributor


class TestInferenceInteraction:
    @classmethod
    def setup_class(cls):
        # Setup method that runs once for the entire test class
        print("\nSetting up the TestCalculator class")

        group1_information = {
            "male": 4,
            "female": 2,
            "age_30_40": 4,
            "age_40_45": 0,
            "age_45_55": 2,
            "age_55_65": 0,
            "age_65_80": 0,
        }

        wrong_group1_information = {
            "malle": 4,
            "female": 2,
            "age_30_40": 4,
            "age_40_45": 0,
            "age_45_55": 2,
            "age_55_65": 0,
            "age_65_80": 0,
        }

        group2_information = {
            "male": 6,
            "female": 2,
            "age_30_40": 6,
            "age_40_45": 0,
            "age_45_55": 2,
            "age_55_65": 0,
            "age_65_80": 0,
        }

        df_g1_as = pd.DataFrame(group1_information, index=[0])
        wrong_df_g1_as = pd.DataFrame(wrong_group1_information, index=[0])
        df_g2_as = pd.DataFrame(group2_information, index=[0])

        sample_data = {
            "sex": "male",
            "age": "age_30_40",
        }
        new_entry = fe_helpers.sample_to_one_hot_encoding(sample_data)

        cls.df_g1_as = df_g1_as
        cls.df_g2_as = df_g2_as
        cls.new_entry = new_entry
        cls.wrong_df_g1_as = wrong_df_g1_as
        cls.total_num_samples = 160

    @classmethod
    def teardown_class(cls):
        # Teardown method that runs once for the entire test class
        print("\nTearing down the TestCalculator class")

        del cls.df_g1_as
        del cls.df_g2_as
        del cls.new_entry
        del cls.total_num_samples

    def setup_method(self):
        # Setup method that runs before each test method
        print("\nSetting up the test method")

        # Instantiate a distributor
        self.distributor = GroupDistributor(
            total_num_samples=self.total_num_samples,
            inference=True,
            df_g1_as=self.df_g1_as,
            df_g2_as=self.df_g2_as,
        )

    def teardown_method(self):
        # Teardown method that runs after each test method
        print("\nTearing down the test method")

    def test_group_assignment(self):
        # result should be group 1
        self.distributor.get_group_update_group_data(self.new_entry) == 1

    def test_feature_names_input(self):
        with pytest.raises(AssertionError):
            self.distributor = GroupDistributor(
                total_num_samples=self.total_num_samples,
                inference=True,
                df_g1_as=self.wrong_df_g1_as,
                df_g2_as=self.df_g2_as,
            )
