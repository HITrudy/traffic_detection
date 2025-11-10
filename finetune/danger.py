import pandas as pd

df = pd.read_csv('/mnt/Project 1_TRACK_POINTS.csv')

df['Time'] = pd.to_datetime(df['Time'].str.replace(' CDT', ''))
df['Time'] = df['Time'].dt.tz_localize('UTC')

time_differences = df['Time'].diff().dt.total_seconds()

acceleration = df['Speed'].diff() / time_differences
acceleration = acceleration.round(2)

jerk = acceleration.diff() / time_differences
jerk = jerk.round(2)

df['Acceleration'] = acceleration
df['Jerk'] = jerk

def is_dangerous(row):
    if abs(row['Acceleration']) > 1 or row['Jerk'] > 0.5:
        return 'Dangerous'
    else:
        return 'Safe'

df['Danger Status'] = df.apply(is_dangerous, axis=1)

def danger_level(row):
    if row['Danger Status'] == 'Safe':
        return 'None'
    elif abs(row['Acceleration']) > 1:
        excess = abs(row['Acceleration']) - 1
        if excess < 2:
            return 'Low'
        elif 2 <= excess < 5:
            return 'Medium'
        else:
            return 'High'
    elif row['Jerk'] > 0.5:
        excess = row['Jerk'] - 0.5
        if excess < 1:
            return 'Low'
        elif 1 <= excess < 3:
            return 'Medium'
        else:
            return 'High'

df['Danger Level'] = df.apply(danger_level, axis=1)

csv_path = '/mnt/Project 1_TRACK_POINTS_with_danger_info.csv'
df.to_csv(csv_path)