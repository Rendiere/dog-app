#!/usr/bin/env bash

pipenv run python -m unittest discover -s "dog_app" --pattern "*_test.py"
