#!/bin/bash

# Run the database check and initialization script
python check_db.py

# Start the FastAPI service in the background
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &

# Start the Streamlit app in the background
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --theme.base="light" &

# Start the scheduler in the foreground
while true
do
  python job.py
  sleep 1800  # Adjust the sleep time as per your requirement
done
