<img src="assets/run.ico" width=70 height=70/> 

# VELOCITI: Can Video Language Models Bind Semantic Concepts Through Time?


## Welcome to the VELOCITI, this repository is to provide code for Evaluation of models on VELOCITI, and provide a jupyter notebook to visualize all the data presented in the benchmark.

## ⭐️ For instant visualization of data samples, please visit our project page: [velociti-benchmark.github.io](velociti-benchmark.github.io)


# Set-Up for Visualizing Data 📊

### Setting-up the Conda Environment
```
conda env create --name velo --file environments/visualizer_env.yaml
conda activate velo
```

### Setting-up Data 💿

- The data is available [Here](https://drive.google.com/file/d/1aKxJL-xv6rS9ChqeLtokKIXBGaRMdD9w/view?usp=sharing) as a `.zip` file.
- Either manually visit the link and donwload the `velociti_data.zip` in the root of this directory, or
- Download the `velociti_data.zip` via `gdown`

```
pip install gdown
cd VELOCITI
```
Then in a python terminal or a file,

```
import gdown
gdown.download('https://drive.google.com/uc?id=1aKxJL-xv6rS9ChqeLtokKIXBGaRMdD9w', 'velociti_data.zip')
```

Unzip the data, and you should see the directory structure as below.

#### Unzip the contents of the `.zip`, and ensure the following directory structure

```
.
├── LICENSE.txt
├── data
│   ├── action_adv.json
│   ├── action_bind.json
│   ├── action_mod.json
│   ├── agent_bind.json
│   ├── agent_iden.json
│   ├── control.json
│   ├── coref.json
│   ├── frames  [900 entries]
│   ├── pos_caps.json
│   ├── sequence.json
│   └── vidsitu_dict.json
└── videos
    ├── velociti_videos_10s  [900 entries]
    └── velociti_videos_4s  [900 entries]
```


Ready for browsing the [provided Jupyter Notebook](data_explore.ipynb) !


# CLIP Model Evaluations

### Environment Setup 🌏

Activate the same environment set-up above,

```
conda activate velo
pip install -r environments/requirements.txt
```

### NOTE 🔔
 
CLIP-ViP model has to be manually downloaded and placed within the folder `.hf_cache` in the root directory of this repository.
Note: All the models, except `CLIP-ViP`, will be automatically downloaded by the script, inside the directory `.hf_cache` in the root folder. The model can be found [here](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP).

If, you wish a different path for the cache files, modify the  `main_eval.py` file accordingly.

After ensuring the above directory structure, and simply run

```
python main_eval.py --num_workers 4 \
                    --all \
                    --output output \ 
                    --exhaustive_log \
                    --seed 1000

```

This will download and run the evaluation on all the models.
If a specific model is to be checked (say `clip_B_32`), then run,

```
python main_eval.py --num_workers 4 \
                    --model clip_B_32 \
                    --output output \ 
                    --exhaustive_log \
                    --seed 1000

```

The `exhasutive_log` saves the output for every sample in the benchmark, and if that level of logging is not required, simply run the evaluation without it, by 


```
python main_eval.py --num_workers 4 \
                    --all \
                    --output output \ 
                    --seed 1000

```


<hr>

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
