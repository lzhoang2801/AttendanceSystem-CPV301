import os
import pandas as pd

ATTENDANCE_PATH = "history/attendance.csv"
os.makedirs(os.path.dirname(ATTENDANCE_PATH), exist_ok=True)

def log_attendance(label_name, date, time):
    if not os.path.exists(ATTENDANCE_PATH):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    else:
        df = pd.read_csv(ATTENDANCE_PATH)

    df.loc[len(df)] = [label_name, date, time]
    df.to_csv(ATTENDANCE_PATH, index=False)

    print(f"Attendance logged for {label_name} at {time}")