#!/bin/bash
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

date +%Y%m%d-%H:%M:%S

# Define project directories
PROJECT_FILES_PATH="${HOME}/project_files"
PROJECT="2020-08-23-genentech"
PROJECT_VERSION="v0-deepmind-original"
INPUT_DATA_DIR="input_data"
MODEL_DATA_DIR="model"
OUTPUT_DATA_DIR="output_data"

INPUT_DATA_PATH="${PROJECT_FILES_PATH}/${PROJECT}/${PROJECT_VERSION}/${INPUT_DATA_DIR}"
MODEL_PATH="${PROJECT_FILES_PATH}/${PROJECT}/${PROJECT_VERSION}/${MODEL_DATA_DIR}"
OUTPUT_DATA_PATH="${PROJECT_FILES_PATH}/${PROJECT}/${PROJECT_VERSION}/${OUTPUT_DATA_DIR}"


# Define input (target) data path
TARGET="T1019s2"  # The name of the target.
TARGET_PATH="${INPUT_DATA_PATH}/${TARGET}"  # Path to the directory with the target input data.
echo "TARGET_PATH=${TARGET_PATH}"


# Define model path
DISTOGRAM_MODEL="${MODEL_PATH}/alphafold_casp13_weights/873731"  # Path to the directory with the distogram model.
BACKGROUND_MODEL="${MODEL_PATH}/alphafold_casp13_weights/916425"  # Path to the directory with the background model.
TORSION_MODEL="${MODEL_PATH}/alphafold_casp13_weights/941521"  # Path to the directory with the torsion model.
echo "DISTOGRAM_MODEL=${DISTOGRAM_MODEL}"
echo "BACKGROUND_MODEL=${BACKGROUND_MODEL}"
echo "TORSION_MODEL=${TORSION_MODEL}"


# Define output path
OUTPUT_DIR="${OUTPUT_DATA_PATH}/contacts_${TARGET}_$(date +%Y_%m_%d_%H_%M_%S)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"


# Set up the virtual environment and install dependencies.
date +%Y%m%d-%H:%M:%S
python3 -m venv alphafold_venv
source alphafold_venv/bin/activate
pip install wheel
pip install -r alphafold_casp13/requirements.txt


# Run contact prediction over 4 replicas.
date +%Y%m%d-%H:%M:%S
for replica in 0 1 2 3; do
  echo "Launching all models for replica ${replica}"

  # Run the distogram model.
  python3 -m alphafold_casp13.contacts \
    --logtostderr \
    --cpu=true \
    --config_path="${DISTOGRAM_MODEL}/${replica}/config.json" \
    --checkpoint_path="${DISTOGRAM_MODEL}/${replica}/tf_graph_data/tf_graph_data.ckpt" \
    --output_path="${OUTPUT_DIR}/distogram/${replica}" \
    --eval_sstable="${TARGET_PATH}/${TARGET}.tfrec" \
    --stats_file="${DISTOGRAM_MODEL}/stats_train_s35.json" &

  # Run the background model.
  python3 -m alphafold_casp13.contacts \
    --logtostderr \
    --cpu=true \
    --config_path="${BACKGROUND_MODEL}/${replica}/config.json" \
    --checkpoint_path="${BACKGROUND_MODEL}/${replica}/tf_graph_data/tf_graph_data.ckpt" \
    --output_path="${OUTPUT_DIR}/background_distogram/${replica}" \
    --eval_sstable="${TARGET_PATH}/${TARGET}.tfrec" \
    --stats_file="${BACKGROUND_MODEL}/stats_train_s35.json" &
done

# Run the torsion model, but only 1 replica.

python3 -m alphafold_casp13.contacts \
  --logtostderr \
  --cpu=true \
  --config_path="${TORSION_MODEL}/0/config.json" \
  --checkpoint_path="${TORSION_MODEL}/0/tf_graph_data/tf_graph_data.ckpt" \
  --output_path="${OUTPUT_DIR}/torsion/0" \
  --eval_sstable="${TARGET_PATH}/${TARGET}.tfrec" \
  --stats_file="${TORSION_MODEL}/stats_train_s35.json" &

date +%Y%m%d-%H:%M:%S
echo "All models running, waiting for them to complete"
wait



# Run the ensembling jobs for distograms, background distograms.
date +%Y%m%d-%H:%M:%S
echo "Ensembling all replica outputs"
for output_dir in "${OUTPUT_DIR}/distogram" "${OUTPUT_DIR}/background_distogram"; do
  pickle_dirs="${output_dir}/0/pickle_files/,${output_dir}/1/pickle_files/,${output_dir}/2/pickle_files/,${output_dir}/3/pickle_files/"

  # Ensemble distograms.
  python3 -m alphafold_casp13.ensemble_contact_maps \
    --logtostderr \
    --pickle_dirs="${pickle_dirs}" \
    --output_dir="${output_dir}/ensemble/"
done

# Only ensemble single replica distogram for torsions.
python3 -m alphafold_casp13.ensemble_contact_maps \
  --logtostderr \
  --pickle_dirs="${OUTPUT_DIR}/torsion/0/pickle_files/" \
  --output_dir="${OUTPUT_DIR}/torsion/ensemble/"

date +%Y%m%d-%H:%M:%S
echo "Pasting contact maps"
python3 -m alphafold_casp13.paste_contact_maps \
  --logtostderr \
  --pickle_input_dir="${OUTPUT_DIR}/distogram/ensemble/" \
  --output_dir="${OUTPUT_DIR}/pasted/" \
  --tfrecord_path="${TARGET_PATH}/${TARGET}.tfrec"

date +%Y%m%d-%H:%M:%S
echo "Done"
