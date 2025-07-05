# Multimodal-and-Multi-label-Decsion-Making
This repository implements a multi-modal learning framework for sleep stage (R, S1, S2, S3, S4, W) and sleep disorder (B, I, N, Na, Nf, P, Rb, S) classification using EEG, ECG, and EMG signals from the PhysioNet CAP Sleep Database (https://physionet.org/content/capslpdb/1.0.0/), as described in Cheng, Y.-H.; Lech, M.; Wilkinson, R.H. Simultaneous Sleep Stage and Sleep Disorder Detection from Multimodal Sensors Using Deep Learning. Sensors 2023, 23, 3468. https://doi.org/10.3390/s23073468. The pipeline processes EDF files to generate spectrograms, trains VGG16 models for each modality, concatenates probability vectors, and uses shallow neural networks (SNN) for final classification.

# Overview

The framework follows a two-level approach:





1. First-Level Models: VGG16 models (using TensorFlow/Keras) are trained on 10-second logarithmic spectrograms for each modality (EEG, ECG, EMG) to classify sleep stages and disorders, producing probability vectors.



2. Second-Level Models: Shallow neural networks combine probability vectors (42-dimensional: 18 for stage, 24 for disorder) to perform final classification.

Key features:





Uses 10-second logarithmic spectrograms (img_channel_10s/log).



Probability vector concatenation (connecting_probability_vectors.py, fully_connect_probability_vectors.py).




Modalities: EEG, ECG, EMG.



Expected accuracies (based on Sensors 2023, 23, 3468):





Sleep stage: ~85-94% (Table 6).



Sleep disorder: ~99% (Table 6).


# Data Source

The data is sourced from the PhysioNet CAP Sleep Database (https://physionet.org/content/capslpdb/1.0.0/). It includes:





EDF files: Polysomnography recordings (e.g., brux1.edf, brux2.edf, ..., sdb4.edf) containing EEG, ECG, EMG, and other signals.



TXT files: Annotation files (e.g., brux1.txt, brux2.txt, ..., sdb4.txt) with sleep stage and disorder labels for 30-second epochs.



Users must download the database and place EDF files in data/ and TXT files in txt_data/


# Prerequisites





Python: 3.8 or higher



Dependencies: Install via pip install -r requirements.txt

Data:





Download EDF files (e.g., brux1.edf, brux2.edf, ..., sdb4.edf) and TXT files (e.g., brux1.txt, brux2.txt, ..., sdb4.txt) from the PhysioNet CAP Sleep Database (https://physionet.org/content/capslpdb/1.0.0/).



Place EDF files in data/ and TXT files in txt_data/.



Ensure db_n.csv contains dynamic range configurations (vmin, vmax) for channels.

# Usage

Follow these steps to run the pipeline:





# Prepare Data:





Download the PhysioNet CAP Sleep Database from https://physionet.org/content/capslpdb/1.0.0/.
EDF files: Polysomnography recordings (108 files, e.g., brux1.edf, brux2.edf, ..., sdb4.edf) containing EEG, ECG, EMG, and other signals.


Place EDF files in data/ and TXT files in txt_data/.



Make sure db_n.csv contains the dynamic range configuration (vmin, vmax) of the channel. You can design your own values.



# Generate Spectrograms:

Use 1_batch_generate_spectrograms.py





Processes EDF files to generate 10-second and 30-second spectrograms.



Outputs: results/spectrograms_example/img_channel_10s/log/ (used for training) and `img_channel_30s/. After this step finished, you should separate the data into folds, train, val, and test.



# Train First-Level VGG16 Models:

Use 3_train_vgg_stage.py
Use 4_train_vgg_disorder.py





Trains VGG16 models (using TensorFlow/Keras) for each modality (EEG, ECG, EMG) for stage and disorder classification.



Inputs: 10-second logarithmic spectrograms (img_channel_10s/log/{modality}/{train,val,test}/{label}/).



Outputs:





Models: results/first_level_model/vgg16_{EEG,ECG,EMG}_{stage,disorder}.h5



Training history: results/first_level_model/hi_df_{EEG,ECG,EMG}_{stage,disorder}.csv



Probability vectors: results/probability_vector/{EEG,ECG,EMG}_{train,val,test}_data_{stage,disorder}.csv



Labels: results/probability_vector/{EEG,ECG,EMG}_{train,val,test}_label_{stage,disorder}.csv





# Concatenate Modality Probability Vectors:

Use 5_connecting_probability_vectors.py





Concatenates EEG, ECG, EMG probability vectors for stage (18D: 6 classes × 3 modalities) and disorder (24D: 8 classes × 3 modalities).



Outputs:





results/probability_vector/{train,test}_data_{stage,disorder}.csv



results/probability_vector/{train,test}_label_{stage,disorder}.csv



Note: Ensure all data is aligned.



# Fully Connect Stage and Disorder Vectors:

Use 6_fully_connect_probability_vectors.py





Concatenates stage and disorder probability vectors into a 42D feature vector (18 stage + 24 disorder).



Outputs:





results/probability_vector/{train,test}_data_combined.csv



results/probability_vector/{train,test}_label_combined_{stage,disorder}.csv



Note: Ensure all data is aligned, especially the sleep disorder and sleep stage data.

# Train Final Shallow Neural Network:

Use 7_SNN_final_decision_maker.py





Trains shallow neural networks (256 → 128 → output) for stage (6 classes) and disorder (8 classes) classification using TensorFlow/Keras.



Inputs: results/probability_vector/{train,test}_data_combined.csv (42D).



Outputs:





Models: results/second_level_model/best_model_{stage,disorder}.h5



Evaluation: Accuracy, confusion matrices, and classification reports for train and test sets.

Note: For sleep stage classification, a pre-trained shallow neural network (PT-Shallow NN) is used to achieve optimal performance, while the sleep disorder model is trained from scratch.

# Validation Data





The pipeline supports validation data ({modality}_val_data_{stage,disorder}.csv), but it is not used in the final SNN training. To include validation data, modify fully_connect_probability_vectors.py to generate val_data_combined.csv and update SNN_final_decision_maker.py to include validation during training.


# Expected Results

Based on Cheng, Y.-H.; Lech, M.; Wilkinson, R.H. Simultaneous Sleep Stage and Sleep Disorder Detection from Multimodal Sensors Using Deep Learning. Sensors 2023, 23, 3468 (Table 6):





Sleep Stage Classification:





Accuracy: 94.34% using a pre-trained shallow neural network (PT-Shallow NN).



Improved by approximately 4% compared to the multimodal-only approach (MML-DMS1).



The trained-from-scratch shallow NN achieved only 84.89% accuracy, necessitating pre-training.



Sleep Disorder Classification:





Accuracy: 99.09% using a shallow neural network trained from scratch.



Improved by approximately 1% compared to the multimodal-only approach (MML-DMS1).



Performance Insights:





The combined multimodal and multilabel approach (MML-DMS2) is more robust to data imbalances across categories compared to single-modality (Experiment 1) and multimodal-only (Experiment 2) approaches.



Sleep stage classification is more challenging than sleep disorder classification, resulting in slightly lower accuracy.



Confusion matrices (Figure 7) show high classification accuracy for individual categories compared to single-modality results (Figure 5).



F1 scores are improved compared to MML-DMS1, indicating better handling of imbalanced data.



Outputs:





Confusion matrices and classification reports for stage (R, S1, S2, S3, S4, W) and disorder (B, I, N, Na, Nf, P, Rb, S) are printed to the console.



Best models saved as results/second_level_model/best_model_{stage,disorder}.h5.

# Citation

If you use this code, please cite:



Cheng, Y.-H.; Lech, M.; Wilkinson, R.H. Simultaneous Sleep Stage and Sleep Disorder Detection from Multimodal Sensors Using Deep Learning. Sensors 2023, 23, 3468. https://doi.org/10.3390/s23073468

# Contact

For issues or questions, please open a GitHub issue or contact the repository owner.
