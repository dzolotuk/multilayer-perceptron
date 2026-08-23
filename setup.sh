#!/bin/bash
python3 -m venv local_env

source local_env/bin/activate

pip install numpy matplotlib pandas

echo "Now use source local_env/bin/activate"