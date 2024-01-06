import re
import urllib.request
import pandas as pd
import numpy as np
import calendar


class NOAAExtractor:

    def __init__(self, url="https://services.swpc.noaa.gov/text/srs.txt"):
        self.url = url
        self.filename = None
        self.month_map = {month: index for index, month in enumerate(calendar.month_abbr) if month}

    @staticmethod
    def location_string_to_lon_lat(loc_str):
        """Converts location string to longitude and latitude values."""
        lat = loc_str[:3].replace('N', '').replace('S', '-')
        lon = loc_str[3:6].replace('W', '').replace('E', '-')
        return int(lon), int(lat)

    @staticmethod
    def convert_to_int(int_str):
        """Converts a string to an integer. Returns np.nan if conversion fails."""
        try:
            return int(int_str)
        except ValueError:
            return np.nan

    def extract_active_regions_from_text(self, content):
        """Extracts active region data from the NOAA content."""
        all_noaa_ars = []

        pattern = r":Issued: (\d{4} \w{3} \d{2} \d{4} UTC)"
        match = re.search(pattern, content)
        if not match:
            print("Pattern not found in the content.")
            return []

        date_time = match.group(1)
        date_time = date_time.strip().split(' ')
        date_time[1] = self.month_map[date_time[1]]
        self.filename = f"{'_'.join(str(item) for item in date_time)}.txt"

        is_plage = False
        for cnt, line in enumerate(content.split('\n')):

            if cnt < 9:  # Skip header lines
                continue

            if line.startswith('I.  Regions with Sunspots.'):
                is_plage = False
                continue
            elif line.startswith('IA. H-alpha Plages without Spots.'):
                is_plage = True
                continue
            elif line.startswith('II. Regions Due to Return'):
                break

            line_values = [attr for attr in line.split(' ') if attr]
            if line_values[0].startswith(('None', 'Nmbr')):
                continue

            year = int(date_time[0])
            month = int(date_time[1])
            day = int(date_time[2])

            noaa_number = int(line_values[0]) + 10000
            lon, lat = self.location_string_to_lon_lat(line_values[1])
            Lo = self.convert_to_int(line_values[2].strip())

            if not is_plage:
                area = int(line_values[3])
                Z = line_values[4]
                LL = int(line_values[5])
                NN = int(line_values[6])
                mag_type = line_values[7].strip()
            else:
                area = np.nan
                Z = 'None'
                LL = np.nan
                NN = np.nan
                mag_type = 'Plage'

            ar_tuple = (year, month, day, noaa_number, lon, lat, Lo, area, Z, LL, NN, mag_type)
            all_noaa_ars.append(ar_tuple)

        return all_noaa_ars

    def fetch_data(self):
        """Fetch content from NOAA."""
        with urllib.request.urlopen(self.url) as fp:
            content = fp.read().decode('utf-8')
        return content

    def get_noaa_dataframe(self):
        """Return the NOAA DataFrame."""
        content = self.fetch_data()
        noaa_ars_data = self.extract_active_regions_from_text(content)
        columns = ['year', 'month', 'day', 'noaa_ar_no', 'longitude', 'latitude', 'carrington_longitude',
                   'corr_whole_spot_area', 'mcintosh', 'LL', 'number_of_spots', 'greenwich']
        noaa_df = pd.DataFrame(data=noaa_ars_data, columns=columns)
        return noaa_df.sort_values(by=['year', 'month', 'day'])
