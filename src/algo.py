import random

import numpy as np
import pandas as pd


class GroupDistributor:
    def __init__(
        self,
        total_num_samples=160,
        inference: bool = False,
        df_g1_as: pd.DataFrame | None = None,
        df_g2_as: pd.DataFrame | None = None,
    ) -> None:
        """
        The Group distributor implement the following algorithm.
        Start with:
            table G1 and G2, where a row represent a sample and a column represents a features
            table G1_summary and G2_summary, where a column represents a features with absolute sum (ac) relative sum (rs)
            table dist_tracker, where row represents a feature, has the following columns
            - importance it is a inverse of a feature probability (the large the number the more rare the feature is so we want to have in both G1 and G2 similar representation)
            - full track whether the feature is fully represented using both groups

        Obtain a sample represented as a row

        Take the distribution tracker and obtain the most important feature where full != 0
            Remark if all features has full == 1, then assign the sample to a random group
        Check if this feature is present in the current sample
            If not take the second most important feature a check its presence.
        Continue until you find a feature that is present in the sample.

        Get from both G1 and G2 the rc values of that feature. Then, compare which rc feature is smaller and
        assign the sample to group with smaller rc feature.

        Compute the feature's total frequency using both groups.
        If the total frequency is same or larger than the expected frequency, then assign  full = 1


        Args:
            total_num_samples (int, optional): Total number of samples to distribute. Defaults to 160.
            inference (bool, optional):
        """
        # Set global variables
        self.target_distribution = {
            "male": 0.4,
            "female": 0.6,
            "age_30_40": 0.05,
            "age_40_45": 0.5,
            "age_45_55": 0.25,
            "age_55_65": 0.4,
            "age_65_80": 0.2,
        }

        self.inference = inference
        self.total_num_samples = total_num_samples
        self.group_features = self.target_distribution.keys()

        if df_g1_as is not None:
            assert set(df_g1_as.columns) == set(
                self.group_features
            ), f"Input should contain exactly only the following columns: {self.group_features}"

            assert set(df_g2_as.columns) == set(
                self.group_features
            ), f"Input should contain exactly only the following columns: {self.group_features}"

            self.df_g1_as = df_g1_as.reset_index(drop=True)
            self.df_g2_as = df_g2_as.reset_index(drop=True)

            self.df_g2_rs = self.df_g1_as
            self.df_g2_rs = self.df_g2_as

            self.update_group_data_rs()

        if self.inference is False:
            self.df_g1 = pd.DataFrame(columns=self.group_features)
            self.df_g1_as = {feat: [0] for feat in self.group_features}
            self.df_g1_as = pd.DataFrame(self.df_g1_as)
            self.df_g1_rs = pd.DataFrame(self.df_g1_as.copy())

            self.df_g2 = pd.DataFrame(columns=self.group_features)
            self.df_g2_as = pd.DataFrame(self.df_g1_as.copy())
            self.df_g2_rs = pd.DataFrame(self.df_g1_as.copy())

        # create the distribution tracker and fill it
        self.dist_tracker_cols = ["importance", "full", "feature"]

        self.imp_list = [round(1 / self.target_distribution[feat], 2) for feat in self.group_features]

        self.dist_tracker = pd.DataFrame(columns=self.dist_tracker_cols)

        self.dist_tracker["importance"] = self.imp_list
        self.dist_tracker["feature"] = self.group_features
        self.dist_tracker["full"] = False

        if self.inference is True:
            for target_feature in self.group_features:
                self.update_full_col_in_dist_tracker_full(target_feature=target_feature)

    def generate_random_sample(self) -> pd.DataFrame:
        data = {}
        for key, probability in self.target_distribution.items():
            data[key] = np.random.choice([0, 1], size=1, p=[1 - probability, probability])[0]

        return pd.DataFrame(data, index=[0])

    def _select_random_group(self, sample_df: pd.DataFrame) -> int:
        """Select random group and update the corresponding group data

        Returns:
            int: randomly selected group
        """
        group = random.choice([1, 2])
        if group == 1:
            self._update_group(1, sample_df)
        else:
            self._update_group(2, sample_df)
        return group

    def _update_group(self, group: int, sample_df: pd.DataFrame) -> None:
        if group == 1:
            if self.inference is False:
                self.df_g1 = pd.concat([self.df_g1, sample_df], axis=0)

            self.df_g1_as = pd.concat([self.df_g1_as, sample_df], axis=0)
            self.df_g1_as = self.df_g1_as.sum(axis=0).to_frame().T
            self.df_g1_rs = self.df_g1_as / self.total_num_samples
            self.df_g1_rs = round(self.df_g1_rs, 2)
        else:
            if self.inference is False:
                self.df_g2 = pd.concat([self.df_g2, sample_df], axis=0)

            self.df_g2_as = pd.concat([self.df_g2_as, sample_df], axis=0)
            self.df_g2_as = self.df_g2_as.sum(axis=0).to_frame().T
            self.df_g2_rs = self.df_g2_as / self.total_num_samples
            self.df_g2_rs = round(self.df_g2_rs, 2)

    def update_full_col_in_dist_tracker_full(self, target_feature: str) -> None:
        """update the colum full for the specified target

        Args:
            target_feature (str): corresponds to the feature
        """
        target_idx = self.dist_tracker[self.dist_tracker["feature"] == target_feature].index[0]
        freq = self.df_g1_as.loc[0, target_feature] + self.df_g2_as.loc[0, target_feature]
        freq = round(freq / self.total_num_samples)
        if self.target_distribution[target_feature] <= freq:
            self.dist_tracker.loc[target_idx, "full"] = 1

    def update_group_data_rs(
        self,
    ) -> None:
        num_samples_g1 = self.df_g1_as.loc[0, "male"] + self.df_g1_as.loc[0, "female"]
        self.df_g1_rs = self.df_g1_as / num_samples_g1
        self.df_g1_rs = round(self.df_g1_rs, 2)

        num_samples_g2 = self.df_g2_as.loc[0, "male"] + self.df_g2_as.loc[0, "female"]
        self.df_g2_rs = self.df_g2_as / num_samples_g2
        self.df_g2_rs = round(self.df_g2_rs, 2)

    def get_group_update_group_data(self, sample_df: pd.DataFrame) -> int:
        """the sample df based on the above algorithm assign the group so that both groups have the same distribution.

        Args:
            sample_df (pd.DataFrame): is Pandas Df with one row, and columns that corresponds to df_g1/df_g2

        Returns:
            int: group assignment
        """

        # assert if all features are available
        assert set(sample_df.columns) == set(
            self.group_features
        ), f"Input should contain exactly only the following columns: {self.group_features}"

        # Check if all feature distributions are filled
        all_full = (self.dist_tracker["full"] == 1).all()
        if all_full:
            group = self._select_random_group(sample_df)

        # sort distribution tracker by importance
        self.dist_tracker = self.dist_tracker.sort_values(by="importance", ascending=False)
        self.dist_tracker = self.dist_tracker.reset_index(drop=True)
        len_dist_tracker = len(self.dist_tracker)
        target_feature = ""
        for idx in range(len_dist_tracker):
            # check if not full
            if self.dist_tracker.loc[idx, "full"] < 1:
                # get the feature
                target_feature = self.dist_tracker.loc[idx, "feature"]

                # check if the feaure is in the sample
                if sample_df.loc[0, target_feature] == 1:
                    break

        # get from both G1 and G2 the rc values of that feature. Then, compare which rc feature is smaller
        # assign the sample to the smaller rc feature. Update the corresponding group
        g1_rc = self.df_g1_rs.loc[0, target_feature]
        g2_rc = self.df_g2_rs.loc[0, target_feature]

        if g1_rc < g2_rc:
            group = 1
            self._update_group(group, sample_df)

        elif g1_rc > g2_rc:
            group = 2
            self._update_group(group, sample_df)
        else:
            group = self._select_random_group(sample_df)

        # compute the feature's total frequency using both groups.
        # If the total frequency is same or larger than the expected frequency, then assign full = 1
        self.update_full_col_in_dist_tracker_full(target_feature=target_feature)

        return group
