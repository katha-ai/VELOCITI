# VERA

To establish that captions in VELOCITI indeed require visual modality and can't simply be solved by a reasoning model, we evaluate and provide script for VERA, a plausability estimation model.

Note: random performance is optimal, as that indicates a struggle to find more plausible choice between positive and negative caption, making the task visually demanding to be solved.

Point the code to the correct data folder with all the `JSON` and the desired output folder, via `data_root` and `outdir` variables, and run


```bash
python vera.py
```