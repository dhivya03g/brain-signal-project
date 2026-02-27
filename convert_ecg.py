import wfdb
import pandas as pd

# Read record 100 from mitdb folder
record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/100')
# Convert signal to DataFrame
df = pd.DataFrame(record.p_signal)

# Save as CSV
df.to_csv('ecg_100.csv', index=False)

print("ECG CSV created successfully!")