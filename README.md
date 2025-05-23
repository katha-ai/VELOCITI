<img src="assets/run.ico" width=70 height=70/> 

# VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment


[![arXiv](https://img.shields.io/badge/arXiv-2406.10889-b31b1b.svg)]([https://arxiv.org/abs/1234.56789](https://arxiv.org/abs/2406.10889))
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/katha-ai-iiith/VELOCITI)


⭐️ For instant visualization of data samples, please visit our [Project Page](https://katha-ai.github.io/projects/velociti/)

### Setting-up the Environment
Create an environment in the choice of your environment manager, and simply install the requirement via
```
cd VELOCITI
# activate your conda or venv environment
pip install -r requirements.txt
```
The code is tested to run with `python 3.11.8`.

### Setting-up Data 💿

The data is available at [HuggingFace](https://huggingface.co/datasets/katha-ai-iiith/VELOCITI).

Use the below command download the entire dataset along with the videos.

```
from huggingface_hub import snapshot_download

snapshot_download(repo_id="katha-ai-iiith/VELOCITI", repo_type='dataset', cache_dir='/path/to/hf/datasets')
```
Check if the videos are downloaded.

```
ls /path/to/hf/datasets/datasets--katha-ai-iiith--VELOCITI/snapshots/*/velociti_videos.zip
```
If you wish to evaluate CLIP style of model, extract the frames using `utils/extract_frames.py`.

```
python utils/extract_frames.py --videos_dir 'velociti_videos' \
                               --frames_dir 'velociti_frames'
```


### Evaluate with VELOCITI

For evaluating with CLIP style of models mentioned in `configs/model_card.py`, run the following:

```
python clip_eval.py --num_workers 0 \
                    --model clip_B_32 \
                    --exhaustive_log \
                    --data_root '/hf/datasets/' \
                    --frames_root 'velociti_frames' \
                    --cache_dir '.cache'
```


For evaluating with Video-LLMs from HuggingFace, run the following:

```
python vidllm_eval.py --frames_root 'velociti_videos' \
                      --cache_dir '/hf/hub/' \
                      --data_root '/hf/datasets/' \
                      --exhaustive_log \
                      --num_workers 0 \
                      --eval_type entail
```
Use `eval_type` argument to swicth between mcq and enatailment evaluation schemes.

The above scripts generate `.csv` files containing the predictions for each sample in the dataset. After this, run `metrics.py` to compute the metrics mentioned in the paper.

The evaluation script for Gemini is `gemini/eval.py`. Note that this may need updates according to the latest Gemini API.

### Consider citing our work if you find it useful.

```
@inproceedings{velociti,
               title={{VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment}},
               author={Saravanan, Darshana and Gupta, Varun and Singh, Darshan and Khan, Zeeshan and Gandhi, Vineet and Tapaswi, Makarand},
               booktitle={Computer Vision and Pattern Recognition (CVPR)},
               year={2025}
   }
```


[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa],

with the following **additional clause** :

🚨 VELOCITI may never be use to tune the parameters of any model, and is strictly a test set.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


