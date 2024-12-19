# Hacking the Encode Function of Sentence Transformers

This repo contains the eight different encode experiments for the Computational Semantics final project.

Each experiment is described in the write-up submitted alongside this repo, and each folder in
this repo contains a single .py file for an experiment. They are self-contained and can simply be run
with:

```
python ./mpnetbase_raw_benchmark/mpnetbase_raw_benchmark.py
```

Assuming all required libraries are installed.

Some dependencies may be system dependent, but in general, the following should be installed
with your preferred package manager (conda or pip) (the latest version for each is fine):

- spacy (and python -m spacy download en_core_web_sm)
- datasets (from HuggingFace)
- pytorch
- transformers
- sentence-transformers

Each file when run will collect the required base models and datasets from HuggingFace, train
the model with any custom encode function, and then test the model. It will save the results locally.

There should be nothing else needed to run this project as each file is a self-contained script,
but please note that these can take about an hour to be trained and evaluated.
