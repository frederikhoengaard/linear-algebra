#!/bin/bash
#
# Perform code style checks of the Python code.

echo "--- black ---"
black --line-length 88 python/main/
echo "--- isort ---"
isort python/main/ --multi-line 3 --profile black
