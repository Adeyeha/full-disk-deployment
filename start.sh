#!/bin/bash

# Start the Streamlit app in the background
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --theme.base="light" &

# Start the scheduler in the foreground
while true
do
  python job.py
  sleep 3600
done
