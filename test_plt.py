import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# CONFIG
CSV_PATH = "reports/daft_model/steer_only/" # Make sure this matches your path
STYLE = 'dark_background' # Looks cool on stream

def find_latest_csv(folder):
    # Find the file with the most recent timestamp
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    
    if not files:      
        try:
            print(files,  max(files, key=os.path.getmtime))
        except Exception as e:
            print(files,  max(files, key=os.path.getmtime))
            print(e)
        return None

    return max(files, key=os.path.getmtime)

def animate(i):
    csv_file = find_latest_csv(CSV_PATH)
    if csv_file is None:
        print("shit, i cannot read")
        exit()
        return
    
    try:
        # Read CSV
        data = pd.read_csv(csv_file)
        print(f"Reading {csv_file}")
        # --- THE FIX IS HERE ---
        # Force 'Success' column to be Numeric (0 or 1)
        # We convert to string first to handle both boolean types and string types safely
        data['Success'] = data['Success'].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0})
        
        # Drop any rows where conversion failed (NaNs) to prevent errors
        data = data.dropna(subset=['Success'])
        # -----------------------

        if len(data) < 2:
            return

        plt.clf()
        fig = plt.gcf()
        fig.suptitle(f"Training Agent: {os.path.basename(csv_file)}", fontsize=10)

        # Plot 1: Reward
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(data['Episode'], data['Total_Reward'], color='#00ff00', linewidth=1)
        ax1.set_title("Total Reward", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Success Rate
        ax2 = plt.subplot(2, 2, 2)
        # Now this will work because it's numbers
        success_rate = data['Success'].rolling(window=100).mean() 
        ax2.plot(data['Episode'], success_rate, color='#00ffff', linewidth=1)
        ax2.set_title("Success Rate (Rolling 100)", fontsize=8)
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Avg Speed
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(data['Episode'], data['Avg_Speed_mps'], color='#ff00ff', linewidth=1)
        ax3.set_title("Avg Speed (m/s)", fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Energy
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(data['Episode'], data['Total_Energy_Wh'], color='#ffff00', linewidth=1)
        ax4.set_title("Energy Consumed (Wh)", fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        
    except Exception as e:
        print(f"Plotting error: {e}")

# Setup Plot
plt.style.use(STYLE)
plt.figure(figsize=(10, 6))

# Update every 5000ms (5 seconds)
ani = FuncAnimation(plt.gcf(), animate, interval=5000)

print("Starting Dashboard... (Keep this window open)")
plt.show()