import os
import sys
from pathlib import Path

project_root_content = "src"
current_folder = os.path.dirname(os.path.realpath(__name__))
while project_root_content not in os.listdir(current_folder):
    current_folder = str(Path(os.path.join(current_folder, "../")).resolve())
sys.path.insert(0, current_folder)

from src.algo import GroupDistributor

# samples
total_num_samples = 160

# instantiate an object
distributor = GroupDistributor(total_num_samples=total_num_samples)

#
print("starting iteration")
for _ in range(total_num_samples):
    random_sample = distributor.generate_random_sample()
    group = distributor.get_group_update_group_data(random_sample)

print("iteration is done")

print("update group data rs")
distributor.update_group_data_rs()

print("group one")
print(distributor.df_g1_as)
print(distributor.df_g1_rs)

print("group two")
print(distributor.df_g2_as)
print(distributor.df_g2_rs)
