import papermill as pm
from pathlib import Path

def smart_execute_notebook(input_notebook_name, output_notebook_name, parameter_dict):
    filepath = Path(output_notebook_name)
    # https://stackoverflow.com/a/35490226/10634604
    # https://stackoverflow.com/a/273227/10634604
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return pm.execute_notebook(input_notebook_name, output_notebook_name, parameter_dict)