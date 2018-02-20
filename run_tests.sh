#!/usr/bin/env bash

pipenv run python -m unittest discover -s "dog-app" --pattern "*_test.py"
