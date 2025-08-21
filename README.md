# ML-Based Query-Driven CardEst Model Used in Calum

## Prepare Data

Download all the files in https://drive.google.com/drive/folders/1xo5kv34YHXY49TShEdNWriW4NxjiSF_9?usp=sharing and execute the following command to decompress them.
```bash
./preprocess/process_data_files.sh
```

## Install Dependencies
We use `uv` to manager the python dependencies. To do so, you need to install `uv`.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
The run the following command to install all packages,
```bash
uv sync
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