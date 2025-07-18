# MASE Triton Examples

## Run the examples

Each folder in this repository contains a different group of examples, and thus requires its own environment setup.
To run the examples,
- Option 1: UV
    The dependencies of each script are already inlined at the top of each script.
    If you have UV installed, you can run the script directly using `uv run <path_to_script>`.

    For example:
    ```bash
    uv run mxfp/03_lm_eval-complex.py --preset_dtype=MXFP4_E2M1 --preset=XWqKV
    ```
- Option 2: Manual
    If you would like to create the environment manually, you can use the `requirements.txt` file in each folder.