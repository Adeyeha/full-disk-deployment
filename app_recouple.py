import streamlit as st
import pandas as pd
import os
import post_hoc_explanation
from post_hoc_analysis import superimpose_original,normalize_image,superimpose_image,superimpose_circular_edge_npy,annotate_points
import numpy as np
import plotly.express as px
from datetime import timedelta


ars = pd.read_csv('ars.csv')
flare_actual = pd.read_csv('Flares.csv')

def get_flare(date_param, df):
    """
    Checks if there are any records in the dataframe where isFlare is True
    and largestEventDate is within the date_param and 24 hours after.

    Parameters:
    date_param (datetime): The date to check against.
    df (DataFrame): The dataframe containing the records.

    Returns:
    DataFrame: A dataframe containing the matching records.
    """
    # Convert the date_param to datetime if it is not already
    if not isinstance(date_param, pd.Timestamp):
        date_param = pd.to_datetime(date_param)

    # Define the end date (24 hours after date_param)
    end_date = date_param + timedelta(hours=24)

    # print(date_param.values[0])
    # print(end_date.values[0])
    # print(type(end_date.values[0]))

    # Filter the dataframe
    result = df[(df['Is_Flare'] == True) & 
                (pd.to_datetime(df['Largest Event Date']) >= date_param.values[0]) & 
                (pd.to_datetime(df['Largest Event Date']) < end_date.values[0])]

    return result

def retrieve_npy(filename,media='media',root=None):
    if root is not None:
        folder = os.path.join(media, root,  *filename.split(' ')[0].split('-'))
        save_path = os.path.join(folder, f"{filename.replace('-','_').replace(' ','_').replace(':','_')}.npy")
        return save_path

def retrieve_ar(request_date,media='media',plage=False,events=False):
    # Ensure request_date is a datetime object
    if not isinstance(request_date, pd.Timestamp):
        try:
            request_date = pd.to_datetime(request_date)
        except Exception as e:
            print(f"Error converting request_date to datetime: {e}")
            return None
    # Filter ars DataFrame for the specific request_date
    ars_select = ars[pd.to_datetime(ars['ar_date']).dt.strftime('%Y-%m-%d') == request_date.date().strftime('%Y-%m-%d')]
    if plage:
        ars_select = ars_select[ars_select['greenwich'] != 'Plage']
    if events:
        ars_select = ars_select[ars_select['Count'].fillna(0) > 0]
    ar_names = ars_select.noaa_ar_no.to_list()
    ar_lat =  ars_select.latitude.to_list()
    ar_lon =  ars_select.longitude.to_list()    
    return ar_names,ar_lat,ar_lon
    
def retrieve_img(filename,media='media',extension='svg',root=None):
    if root is not None:
        folder = os.path.join(media, root,  *filename.split(' ')[0].split('-'))
        save_path = os.path.join(folder, f"{filename.replace('-','_').replace(' ','_').replace(':','_')}.{extension}")
        return save_path

def main():
    """
    Streamlit app
    """
    # st.title("Full Disk Solar Flare Predictor")
    # st.write("Welcome to our Solar Flare Prediction System, utilizing a deep-learning model trained on full-disk magnetogram images, we issue an hourly forecast of ≥M-class solar flares with a 24-hour prediction window. With a full-disk approach, our model extends its prediction capabilities to encompass near-limb events. We employ Guided Grad-CAM for interpretability, providing post hoc explanations for predictions and evaluate the explanations regarding the location of active regions. Explore reliable and data-driven solar flare forecasts with our system.")
    st.markdown("""
    <div style="text-align: center;">
        <h1>Full Disk Solar Flare Predictor</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify;">
        Welcome to our Solar Flare Prediction System, utilizing a deep-learning model trained on full-disk magnetogram images, we issue an hourly forecast of ≥M-class solar flares with a 24-hour prediction window. With a full-disk approach, our model extends its prediction capabilities to encompass near-limb events. We employ Guided Grad-CAM for interpretability, providing post hoc explanations for predictions and evaluate the explanations regarding the location of active regions. Explore reliable and data-driven solar flare forecasts with our system.
    </div>
    <br>
    """, unsafe_allow_html=True)

    st.sidebar.title("Flare Information")

    db_file = "predictions.txt"
    db = pd.read_csv(db_file)

    # selected_date = st.sidebar.date_input("Past Predictions Date").strftime("%Y-%m-%d")

    # # db = db[db['obs_date'].startswith(selected_date)].sort_values('obs_date',ascending=True)
    # print(db.columns)
    # # Filter the DataFrame based on the condition and sort by 'obs_date'
    # db = db[db['obs_date'].str.startswith(selected_date)].sort_values('obs_date', ascending=True)

    if db.empty:

        st.markdown("""
            <div style="text-align: center;">
                <strong>No data to display</strong>
            </div>
            <br>
            """, unsafe_allow_html=True)

    else:
        # Convert the 'timestamp' column to a list of strings
        timestamps_list = sorted(list(set(db['obs_date'].to_list())),reverse=True)

        selected_date = st.sidebar.selectbox(
            f"Past Predictions ", options=timestamps_list
        )

        # Configure parameters
        st.sidebar.title("Filter ARs")
        # Create a toggle button
        plage = st.sidebar.checkbox("Plage", value=False)
        events = st.sidebar.checkbox("Event", value=False)

        # Configure parameters
        st.sidebar.title("Boundary Configuration")
        angle = st.sidebar.slider("Angle", min_value=5, max_value=90, value=20)
        es_buffer = st.sidebar.slider("ES Buffer", min_value=2, max_value=100, value=5)
        ws_buffer = st.sidebar.slider("WS Buffer", min_value=5, max_value=100, value=40)

        st.sidebar.title("AR Detection Configuration")
        lower_threshold = st.sidebar.slider("Lower Threshold", min_value=10, max_value=255, value=30)
        upper_threshold = st.sidebar.slider("Higher Threshold", min_value=10, max_value=255, value=50)

        st.sidebar.title("Density Configuration")
        min_samples = st.sidebar.slider("Min. Samples", min_value=1, max_value=10, value=2)
        distance_threshold = st.sidebar.slider("Distance Threshold", min_value=1, max_value=20, value=10)

        if int(ws_buffer) - int(es_buffer) >= 5:


            nw_angle = -angle
            sw_angle =  angle

            # Find the row with the specific timestamp in the DataFrame
            row = db.loc[db['obs_date'] == selected_date].sort_values('local_request_date',ascending=False).head(1)
            guidedgradcam = retrieve_npy(row['local_request_date'].values[0],media='media',root='guidedgradcam')
            original = retrieve_npy(row['local_request_date'].values[0],media='media',root='original')
            ar_names,ar_lat,ar_lon = retrieve_ar(row['local_request_date'].values[0],
            plage=plage ,
            events=events,
            media='media'

            )

            actual_pred = get_flare(row['obs_date'],flare_actual)
            # # Parse flare information
            # ar_names = ar.noaa_ar_no.to_list()
            # ar_lat =  ar.latitude.to_list()
            # ar_lon =  ar.longitude.to_list()

            pix_list = [post_hoc_explanation.convert_to_pix(float(x[1]),float(x[0])) for x in zip(ar_lat,ar_lon)]
            flares = [item for item in list(zip(ar_names,pix_list))]

            # Analyze attention map
            bounding_hulls_img, distances_df, score, ratio = post_hoc_explanation.explain(guidedgradcam, flares, nw_angle, sw_angle, es_buffer, ws_buffer,lower_threshold,upper_threshold,min_samples,distance_threshold, numpy=True)

            print(nw_angle, sw_angle, es_buffer, ws_buffer,lower_threshold,upper_threshold,min_samples,distance_threshold)

            col1, col2 = st.columns(2)
            col1.image(np.load(original), caption=f"Input Magnetogram @ {selected_date}UTC", use_column_width=True)

            # Superimposed Image - Original & Attention
            superimposed = superimpose_original(np.load(original), np.load(guidedgradcam))

            col1.image(superimposed, caption="Superimposed Post hoc Explanation", use_column_width=True)

            col2.image(superimpose_original(np.load(original), np.load(guidedgradcam),1),caption="Post hoc Explanation Map (Guided Grad-CAM)", use_column_width=True)

            # col2.image(bounding_hulls_img,caption="Explained Image", use_column_width=True)

            # col1.image(superimpose_image(np.load(original), bounding_hulls_img),caption="Explained Image", use_column_width=True)

            col2.image(annotate_points(distances_df,flares,superimpose_circular_edge_npy(original, bounding_hulls_img)),caption="Evaluation of Explanation with AR Location", use_column_width=True)
            

            col1, col2 = st.columns(2)
            st.write('\n\n')
            col2.write(distances_df.rename(columns = {'Flare':'NOAA AR','Distance':'Distance (in pixels)'}))
            # col1.write(f"Flare Probability (≥ M1.0) = {round((row['flare_probability'].values[0] * 100),2) }%")
            # col1.markdown(f"**Flare Probability (≥ M1.0) = {round((row['flare_probability'].values[0] * 100),2)}%**")
            col1.markdown(f'<span style="color: blue; font-size: larger;"><b>Flare Probability (≥ M1.0) = {round((row["flare_probability"].values[0] * 100), 2)}%</b></span>', unsafe_allow_html=True)
            col1.write(f"Proximity Score = {score}")
            col1.write(f"Collocation Ratio = {round(ratio,2) * 100}%")
            if actual_pred.shape[0] > 1:
                col1.write(f"Actual = Flare")
            else:
                col1.write(f"Actual = Non Flare")            
            st.markdown("""
                <div style="text-align: justify;">
                <strong>Distance:</strong> shows whether the AR location is inside the bounding contour of the explanation or outside. The value is  zero if it lies with in the bounding contour. Otherwise it's the absolute difference in distance (in terms of pixels) between the AR Location and the nearest edge of the bounding region.
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: justify;">
                <strong>Proximity Score:</strong> For distances <strong>D<sub>i</sub></strong> between (n) number of AR locations <strong>A<sub>i</sub></strong> and nearest edge of the respective bounding contour,
                <br>
                Proximity score: &sum;<sub>i=1</sub><sup>n</sup> D<sub>i</sub> / n
                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: justify;">
                <strong>Collocation Ratio :</strong> For <strong>X</strong> number of total NOAA ARs present in the line-of-sight full-disk input magnetogram, if <strong>Y</strong> (which is: <strong>Y&le;X</strong>) numbers of ARs are contained within the hulls of the explanations, then, collocation score: <strong>Y/X</strong>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.error("Error: Westward Buffer must be larger than Eastward Buffer by at least 5 pixels")

if __name__ == "__main__":
    main()