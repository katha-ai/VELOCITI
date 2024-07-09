# mPLUG-Owl-Video and Owl-Con Evaluation

We provide scripts to facilitate inference on the models mPLUG-Owl-Video and Owl-Con.

Please refer to the [original repo](https://github.com/Hritikbansal/videocon) for the model, core code components and the environment setup.

For the inference, follow the instructions provided in the [Entailment-Inference](https://github.com/Hritikbansal/videocon?tab=readme-ov-file#entailment-inference) section.

- Skip steps 1 and 2.
- Instead run `gen_input_csv.py` to obtain the input_csv files for any of the tests.
    ```
    python gen_input_csv.py --test ag_iden
    ```
- In step 3, pass the generated input csv using the `--input_csv` parameter

    ```
    CUDA_VISIBLE_DEVICES=0 python entailment_inference.py --input_csv videocon_prompts/ag_iden.csv --output_csv ../../examples/final_test_scores.csv --trained_ckpt <path to pytorch.bin of videocon ckpt> --pretrained_ckpt <path to mplugowl-7b-video folder> --use_lora --all-params
    ```
- Follow step 4 for inference on mPLUG-Owl-Video
- Use the functions available in `calculate_metrics.py` to compute accuracy and other metrics mentioned in the paper.