# PLLaVA Model Evaluation

We provide scripts to facilitate inference on the PLLaVA model.

Please refer to the [original repo](https://github.com/magic-research/PLLaVA) for the model, core code components and the environment setup.

Note: The [model utils file](https://github.com/magic-research/PLLaVA/blob/main/tasks/eval/model_utils.py) has been slightly modified to accomodate entailment score calculation. We present the updated file `model_utils.py`. At the time of the writing the code, simply replacing the original file with the provided one was working. However, updates made to the original repository may affect this.

The key components of the new `model_utils.py` are: to calculate the entailment score from the `model.generate()` call.
```python
def get_score(logits, token_id_yes, token_id_no):
    logits = F.softmax(logits, dim=-1)
    score = logits[token_id_yes] / (logits[token_id_yes] + logits[token_id_no])
    return score.item(), logits[token_id_yes].item(), logits[token_id_no].item()
```
and some updates to the `pllava_answer` function.

Once the data directory is set up, modify the followig args (within `evaluate_velo.py`), in either terminal or the code:

```python
parser.add_argument("--pretrained_model_name_or_path", type=str, default="MODELS/pllava-7b") #path to the model directory, as set up from the original PLLaVA repo.
parser.add_argument("--weight_dir", type=str, default="MODELS/pllava-7b")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--videos_dir", type=str, default="data/videos/velociti_videos_10s")
parser.add_argument("--videos_dir_4s", type=str, default="data/videos/velociti_videos_4s")
parser.add_argument("--outdir", type=str, default="RESULTS") # where result csv's will be saved.
parser.add_argument("--data_root", type=str, default="data")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=1000)
```

then, simply run;

```bash
python evaluate_velcro.py
```

this will save various `CSVs` in the `outdir` folder.

Next, to calculate the metrics,

simply run,

```bash
python calculate_metrics.py
```

Note: `calculate_metrics.py` expects the `CSVs` to be in an folder named `RESULTS`, if the outdir was changed, update the code accordingly.
