# KG-Completion

# Preprocessing (make KG)
python preprocessing.py

# Training
cuda_visible_devices="devices" python main.py

# Evaluation
cuda_visible_devices="devices" python predict_main.py --model-name 'name'