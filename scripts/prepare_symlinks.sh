#!/bin/bash

set errexit
set nounset

BVI_DVC=/homes/datasets/BVI_DVC
DATA_DIR=/homes/students_home/lorenzopalloni/Projects/binarization/data

# remove symlinks if they exist
if [[ -L "${DATA_DIR}/original_videos" ]]; then
    rm "${DATA_DIR}/original_videos"
fi
if [[ -L "${DATA_DIR}/compressed_videos" ]]; then
    rm "${DATA_DIR}/compressed_videos"
fi
if [[ -L "${DATA_DIR}/original_frames" ]]; then
    rm "${DATA_DIR}/original_frames"
fi
if [[ -L "${DATA_DIR}/compressed_frames" ]]; then
    rm "${DATA_DIR}/compressed_frames"
fi

ln -s "${BVI_DVC}/3hj4t64fkbrgn2ghwp9en4vhtn/Videos/" "${DATA_DIR}/original_videos"
ln -s "${BVI_DVC}/previous_data/encoded/encoded_QF23/" "${DATA_DIR}/compressed_videos"
ln -s "${BVI_DVC}/previous_data/frames/frames_HQ/" "${DATA_DIR}/original_frames"
ln -s "${BVI_DVC}/previous_data/frames/frames_JPG_QF23/" "${DATA_DIR}/compressed_frames"

