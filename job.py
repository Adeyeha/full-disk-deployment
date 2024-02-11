from deployment import FullDiskFlarePrediction
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')
import sqlite3
from dotenv import load_dotenv
load_dotenv()

def write_to_csv(result, csv_filename='results.csv'):
    df = pd.DataFrame([result])
    df.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)

def make_predictions(save_artefacts=True, include_explain=False):
    PATH1 = os.getenv('model_path')
    fdp = FullDiskFlarePrediction(PATH1)
    return fdp.predict(save_artefacts=save_artefacts, include_explain=include_explain)

def write_to_db(prediction, latest=False):
    db_name = os.getenv('db_name')
    table_name = os.getenv('most_recent_record') if latest else os.getenv('all_records')
    
    with sqlite3.connect(db_name) as conn:
        cur = conn.cursor()
        if latest:
            cur.execute(f'DELETE FROM {table_name}')  # Clear the table for latest record
            
        cur.execute(f'''
        INSERT INTO {table_name} (source_date, obs_date, raw_filename, noaa_ar_filename, local_request_date, error, flare_probability, non_flare_probability, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction['source_date'],
            prediction['obs_date'],
            prediction['raw_filename'],
            prediction['noaa_ar_filename'],
            prediction['local_request_date'],
            prediction['error'],
            float(prediction['flare_probability']),
            float(prediction['non_flare_probability']),
            prediction['explanation']
        ))
        conn.commit()

def main():
    try:
        prediction = make_predictions(save_artefacts=True, include_explain=False)
        write_to_csv(prediction)
        write_to_db(prediction)
        write_to_db(prediction, latest=True)

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()
    print('processed')
