import pandas as pd


def sample_to_one_hot_encoding(sample_data: dict) -> pd.DataFrame:
    """

    Args:
        group_infromation (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    sample_one_hot_enc = dict()

    if sample_data["sex"] == "male":
        sample_one_hot_enc["male"] = [1]
        sample_one_hot_enc["female"] = [0]
    elif sample_data["sex"] == "female":
        sample_one_hot_enc["male"] = [0]
        sample_one_hot_enc["female"] = [1]
    # Assuming "age" is always selected, and using direct mapping
    for age_group in ["age_30_40", "age_40_45", "age_45_55", "age_55_65", "age_65_80"]:
        sample_one_hot_enc[age_group] = [1] if sample_data["age"] == age_group else [0]

    sample_one_hot_enc = pd.DataFrame(sample_one_hot_enc)

    return sample_one_hot_enc
