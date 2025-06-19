import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time

# Get the project root directory and set up data path
script_dir = os.path.dirname(__file__)
project_root = os.path.join(script_dir, '..', '..')
data_csv_dir = os.path.join(project_root, 'data', 'csv')

FILE = Path(os.path.join(data_csv_dir, 'covariance_data.csv'))
PARAMS = [
    'Number of Jumps','Tag_Est_X','Tag_Est_Y','Tag_Est_Z',
    'Tag_Est_Roll','Tag_Est_Pitch','Tag_Est_Yaw'
]

# Covariance bar chart setup
plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(PARAMS, [0]*len(PARAMS))
ax.set_ylabel('Covariance with Translation Error')
ax.set_title('Covariance between Parameters and Translation Error')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Additional figure for error plot over readings as a point graph
fig2, ax2 = plt.subplots()
line, = ax2.plot([], [], marker='o', linestyle='None')
ax2.set_xlabel('Reading #')
ax2.set_ylabel('Translation Error')
ax2.set_title('Translation Error over Readings')
plt.tight_layout()

last_mtime = 0
while plt.fignum_exists(fig.number) and plt.fignum_exists(fig2.number):
    try:
        mtime = FILE.stat().st_mtime
        if mtime != last_mtime:  # update only when file changes
            last_mtime = mtime
            df = pd.read_csv(FILE)

            # Update covariance bar chart
            cov = df.cov()['Translation_Error'].reindex(PARAMS)
            for bar, val in zip(bars, cov):
                bar.set_height(val)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            
            # Update error over readings point plot
            if 'Translation_Error' in df.columns:
                x_values = list(range(len(df)))
                y_values = df['Translation_Error'].tolist()
                line.set_data(x_values, y_values)
                ax2.relim()
                ax2.autoscale_view()
                fig2.canvas.draw_idle()

        plt.pause(0.1)  # GUI heartbeat
    except FileNotFoundError:
        print('Waiting for file…')
        time.sleep(1)
    except Exception as e:
        print('Read failed:', e)
        time.sleep(0.5)
