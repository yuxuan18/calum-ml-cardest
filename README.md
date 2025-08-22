# ML-Based Query-Driven CardEst Model Used in Calum

## Install Dependencies
We use `uv` to manager the python dependencies. To do so, you need to install `uv`.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
The run the following command to install all packages,
```bash
uv sync 
```

## Prepare Data

To start from raw data, first download raw query plans features from https://bytedance.larkoffice.com/drive/folder/GylefZMK1lPKaddbyuQcUQB6nLe and decompress all the files. Then, just execute the following command to do the convert raw query plans into processed data for training and testing.
```bash
# preprocess data for tpch
mkdir -p data/tpch
uv run preprocess/tpch_data_convert.py --input_dir tpch_1t_plans/ --output_dir data/tpch --mode train

# preprocess data for tpcds
mkdir -p data/tpcds
uv run preprocess/tpcds_data_convert.py --input_dir tpcds_1t_plans/ --output_dir data/tpcds --mode train

# proprecess data for devmind
mkdir -p data/devmind
uv run preprocess/tpcds_data_convert.py --input_dir tpcds_1t_plans/ --output_dir data/tpcds --mode train
```
---

To start from processed data and trained mode, download all the files in https://bytedance.larkoffice.com/drive/folder/QcwPfHoguljrA5dhtZ8ccBpAn3g and https://bytedance.larkoffice.com/drive/folder/DbWrfNPJLl5ELPdGorHchE0Unjc and execute the following command to decompress them.
```bash
./preprocess/process_data_files.sh
```

## Train or Test Models

### Zero-Shot

To the train or test the zero-shot model, first `cd` to the model's directory:
```bash
cd src/zs-devmind # train a model for devmind
cd src/zs-tpcds # train a model for tpcds
cd src/zs-tpch # train a model for tpch
```

Then, run the following command
```bash
uv run train.py --train_model \
--workload_runs {train_data} \
--test_workload_runs {test_data} \
--statistics_file {meta_data} \
--target {directory_to_save_model} \
--filename_model {name_of_saved_model} \
--hyperparameter_path setup/tuned_hyperparameters/tune_est_card_config.json \
--max_epoch_tuples 100000 \
--loss_class_name QLoss \
--device cpu \
--num_workers 32 \
--database postgres \
--plan_featurization KryptonMultiCardDetail \
--seed 0
```
For devmind, use
```bash
--workload_runs ../../data/devmind/devmind_train_data.json \
--test_workload_runs ../../data/devmind/devmind_test_data.json \
--statistics_file ../../data/devmind/statistics.json \
--target ../../models/devmind \
--filename_model devmind_card \
```
For tpch, use
```bash
--workload_runs ../../data/tpch/tpch_train_data.json \
--test_workload_runs ../../data/tpch/tpch_test_data.json \
--statistics_file ../../data/tpch/statistics.json \
--target ../../models/tpch \
--filename_model tpch_card \
```
For tpcds, use
```bash
--workload_runs ../../data/tpcds/tpcds_train_data.json \
--test_workload_runs ../../data/tpcds/tpcds_test_data.json \
--statistics_file ../../data/tpcds/statistics.json \
--target ../../models/tpcds \
--filename_model tpcds_card \
```

### GRASP
To train or test GRASP, first `cd src/grasp`. Then, use the following commands 

to test tpch:
```bash
uv run train_grasp_tpch.py --data_path queries/tpch.csv --test_only
```

to test tpcds:
```bash
uv run train_grasp_tpcds.py --data_path queries/tpcds.csv --test_only
```

To train from scratch, remove the `--test_only` flag. Note that this will overwrite the previously trained models.

### MSCN
To train or test MSCN, first `cd src/mscn`. Then, use the following commands

to train tpcds:
```bash
uv run train.py tpch --epoch 10000 --dataset tpcds
```

to test tpcds:
```bash
uv run train.py tpch_test --epoch 10000 --test_only --dataset tpcds
```

to train tpch:
```bash
uv run train.py tpch --epoch 10000 --dataset tpch
```

to test tpch:
```bash
uv run train.py tpch_test --epoch 10000 --test_only --dataset tpch
```

## Calculate Uncertainty and Threshold

Prediction files (`.csv`) and embedding files (`.npy`) should be prepared before calculating the uncertainty.
1. A prediction file is a CSV file with two columns: ['truecard', 'predcard']. Note that the prediction file should not contain a header.
2. An embedding file is a NPY file containing a list of embedding vectors.

Both files can be obtained by running the zero-shot model.

To get uncertainty quantification of test data, run the following command with appropriate parameters.
```bash
uv run src/calculate_uncertainty.py --mode uncertainty \
--train_prediction_file {prediction_file_of_training_data} \
--train_embedding_file {embedding_file_of_training_data} \
--test_embedding_file {embedding_file_of_test_data} \
--test_uncertainty_file {output_file} \
--k_ratio 0.0005
```
`k_ratio` defines the number of nearest neighbor to consider according to a specific ratio of the training data. We suggest using 0.0005 for a large training dataset (>10k) and 0.1 for a small training dataset (~1k)

To get a threshold with Q-error interpretations, run the following command with appropriate parameters
```bash
uv run src/calculate_uncertainty.py --mode threshold \
--train_prediction_file {prediction_file_of_training_data} \
--train_embedding_file {embedding_file_of_training_data} \
--test_embedding_file {embedding_file_of_test_data} \
--test_uncertainty_file {output_file} \
--k_ratio 0.0005 \
--validation_ratio 0.2 \
--max_qerror 25 \
--target_recall 0.8
```
`validation_ratio` defines the number of data used for calibrate the threshold. We suggest using 0.2 for a large training dataset (>10k) and 0.8 for a small training dataset (~1k).
