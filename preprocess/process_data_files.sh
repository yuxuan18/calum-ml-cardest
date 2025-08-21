#!/bin/bash

mkdir -p models
tar -xzvf zs_trained_models.tar.gz

mkdir -p data
tar -xzvf tpch_processed_data.tar.gz -C data/
tar -xzvf tpcds_processed_data.tar.gz -C data/
tar -xzvf devmind_processed_data.tar.gz -C data/

tar -xzvf grasp_trained_models.tar.gz -C src/grasp/
tar -xzvf mscn_trained_models.tar.gz -C src/mscn/