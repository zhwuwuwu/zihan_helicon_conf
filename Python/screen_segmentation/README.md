# Env setup

1. conda create -n screen_segmentation python=3.13.3

2. pip install -r requirements.txt

# Data preprare
python dataset_prepare.py

# Train
python train.py

# Evaluate
python evaluate.py

# Simple test
python infer_test.py 