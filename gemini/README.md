# Gemini Evaluation

We provide the script to evaluate Gemini 1.5 Flash on the subset of the data that was used for Human evaluation.

The data is available in the `gemini/gemini_data` folder.

Run the command below to evaluate the model on any test. Follow the instructions in the [Gemini-API](https://ai.google.dev/gemini-api/docs/api-key) page to obtain the API key.

```bash
python eval.py --test ag_iden --gemini_gcp_key <KEY>
```

