#!/bin/bash

# execute from witin src/
source /workspaces/venv/bin/activate
uvicorn rest_api.api:app --port 8500 &
python3 ui/webapp.py
