import base64
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root_content = "src"
current_folder = os.path.dirname(os.path.realpath(__name__))
while project_root_content not in os.listdir(current_folder):
    current_folder = str(Path(os.path.join(current_folder, "../")).resolve())
sys.path.insert(0, current_folder)

import src.frontend.helpers as fe_helpers
from src.algo import GroupDistributor

# Prepare working environment
# samples
total_num_samples = 160

# Streamlit app
st.title("Group Assignment App")

st.write("Enter Group 1 Information:")
group1_information = {
    "male": st.selectbox("male:", list(range(total_num_samples))),
    "female": st.selectbox("female:", list(range(total_num_samples))),
    "age_30_40": st.selectbox("age_30_40:", list(range(total_num_samples))),
    "age_40_45": st.selectbox("age_40_45:", list(range(total_num_samples))),
    "age_45_55": st.selectbox("age_45_55:", list(range(total_num_samples))),
    "age_55_65": st.selectbox("age_55_65:", list(range(total_num_samples))),
    "age_65_80": st.selectbox("age_65_80:", list(range(total_num_samples))),
}

st.write("-----")
st.write("Enter Group 2 Information:")
group2_information = {
    "male": st.selectbox("2male:", list(range(total_num_samples))),
    "female": st.selectbox("2female:", list(range(total_num_samples))),
    "age_30_40": st.selectbox("2age_30_40:", list(range(total_num_samples))),
    "age_40_45": st.selectbox("2age_40_45:", list(range(total_num_samples))),
    "age_45_55": st.selectbox("2age_45_55:", list(range(total_num_samples))),
    "age_55_65": st.selectbox("2age_55_65:", list(range(total_num_samples))),
    "age_65_80": st.selectbox("2age_65_80:", list(range(total_num_samples))),
}

df_g1_as = pd.DataFrame(group1_information, index=[0])
df_g2_as = pd.DataFrame(group2_information, index=[0])

# instantiate an object
distributor = GroupDistributor(
    total_num_samples=total_num_samples,
    inference=True,
    df_g1_as=df_g1_as,
    df_g2_as=df_g2_as,
)

# Show distributions of both groups
st.write("Summary statistics of Group 1:")
st.write(distributor.df_g1_as)
st.write(distributor.df_g1_rs)

st.write("Summary statistics of Group 2:")
st.write(distributor.df_g2_as)
st.write(distributor.df_g2_rs)

# Widget to enter data of a new sample and then show sample
st.header("Section 2: Enter Data and Show Summary")

sample_data = {
    "sex": st.selectbox("sex:", ["male", "female"]),
    "age": st.selectbox("age:", ["age_30_40", "age_40_45", "age_45_55", "age_55_65", "age_65_80"]),
}
new_entry = fe_helpers.sample_to_one_hot_encoding(sample_data)

# Show summary
st.write("Summary of Entered Data:")
st.write(new_entry)

# Assign group and show summary statistics offer download button to get complete statistics back
if st.button("Submit Sample To Get Group"):
    # Group assignment logic (you can replace this with your own logic)
    group_assignment = distributor.get_group_update_group_data(new_entry)

    st.header("Section 3: Group Assignment and Summary Statistics")
    st.write(f"The person should be assigned to {group_assignment}")

    st.write("Updated summary statistics of Group 1:")
    st.write(distributor.df_g1_as)
    st.write(distributor.df_g1_rs)

    st.write("Updated summary statistics of Group 2:")
    st.write(distributor.df_g2_as)
    st.write(distributor.df_g2_rs)

    # Offer to download data
    now = dt.datetime.now().strftime("%d%m%Y-%H%M")
    file_name = f"group_summaries_{now}"

    # Function to create a download link
    def download_link(df, file_name, label):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64 encoding
        href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">{label}</a>'
        return href

    # Download buttons
    st.markdown(download_link(distributor.df_g1_as, f"g1_as_{file_name}", "Download g1 as"), unsafe_allow_html=True)
    st.markdown(download_link(distributor.df_g1_rs, f"g1_rs_{file_name}", "Download g1 rs"), unsafe_allow_html=True)

    st.markdown(download_link(distributor.df_g2_as, f"g2_as_{file_name}", "Download g2 as"), unsafe_allow_html=True)
    st.markdown(download_link(distributor.df_g2_rs, f"g2_rs_{file_name}", "Download g2 rs"), unsafe_allow_html=True)
