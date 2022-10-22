#!/bin/bash

pkill -u lorenzopalloni gunicorn

MLFLOW_BACKEND_STORE_URI=file:///homes/students_home/lorenzopalloni/Projects/binarization/artifacts/mlruns

mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --host 127.0.0.1 --port 5005

