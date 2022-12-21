import os
"""
youtube model package params
load and validate the environment variables in the `.env`
"""

DATASET = "small" # ["small","lunch3", "lunch4", "finalesterest", "merge"]
TABULAR = False
LOG = True
BUCKET_NAME = os.environ.get("BUCKET_NAME")
TIMESTAMP = "20221220-232504" # ["20221204-144654"]
