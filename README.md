# group_distributor
Algorithm with with a simple Frontend to distribute a sample in one of two groups.


## Setup dev enviroment using Miniconda
1. Install Miniconda
2. Execute step-by-step the following commands in your terminal
```
conda env create --file ./env-configs/dev-env/environment.yml
conda activate group_dist_dev-env
pip install -r ./env-configs/dev-env/requirements.txt
```

3. If VS Code is used as an IDE, then type in the extension market place `@recommended` to install extensions.

4. Start the application:
```
streamlit run src/frontend/main.py
```
It should automatically open a web-browser. Now, you can interactively add a sample and obtain an appropriate group.
