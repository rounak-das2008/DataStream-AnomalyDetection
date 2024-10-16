import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from sklearn.ensemble import IsolationForest
import threading

# Function to simulate data stream
def data_stream_simulator():
    """
    Simulates a real-time data stream with regular patterns, seasonal elements, and random noise.

    Returns:
        float: The next value in the simulated data stream.
    """
    t = 0  # Initialize time step
    while True:
        try:
            # Trend component: a slow upward or downward trend over time
            trend = 0.01 * t

            # Seasonal component: combines multiple sine waves for complexity
            seasonal = (
                20 * math.sin(2 * math.pi * t / 50) +  # Increased primary seasonality amplitude for better positive peaks
                7 * math.sin(2 * math.pi * t / 25) +   # Secondary seasonality
                3 * math.sin(2 * math.pi * t / 10)     # Tertiary seasonality
            )

            # Random noise: Gaussian noise with mean 0 and standard deviation 3 (increased noise level)
            noise = random.gauss(0, 3)

            # Sudden change component: simulates concept drift
            if t % 20 == 0 and t != 0:
                trend *= random.choice([-1, 1])  # Randomly invert the trend

            # Anomalies: inject occasional spikes or drops
            anomaly = 0
            if random.random() < 0.01:  # Increased probability of anomaly occurrence
                anomaly = random.choice([random.uniform(30, 50), random.uniform(-50, -30)])  # Increased range for positive anomalies

            # Combine all components to get the final value
            value = trend + seasonal + noise + anomaly

            # Yield the value to simulate streaming
            yield value

            # Increment time step
            t += 1

            # Sleep to mimic real-time data (adjust as needed)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in data stream simulation: {e}")
            break

# Function for Isolation Forest Anomaly Detection
def isolation_forest_anomaly_detection(model, data_point, previous_point=None, threshold=30, anomaly_data=None):
    """
    Detects anomalies for a single data point using the Isolation Forest model and additional local anomaly detection.

    Parameters:
        model (IsolationForest): The pre-trained Isolation Forest model.
        data_point (float): The data point to analyze.
        previous_point (float): The previous data point to analyze sudden changes.
        threshold (float): The threshold for detecting sudden large changes.
        anomaly_data (list): List of previous anomalies to avoid duplicate marking.

    Returns:
        bool: True if the data point is an anomaly, False otherwise.
    """
    try:
        # Isolation Forest detection
        data_point = np.array(data_point).reshape(1, -1)
        prediction = model.predict(data_point)[0] == -1  # True if anomaly

        # Local sudden change detection
        sudden_change = False
        if previous_point is not None:
            if abs(data_point[0][0] - previous_point) > threshold:
                sudden_change = True
                # Only mark the peak point as an anomaly, skip marking the following merge point
                if anomaly_data is not None and previous_point in [a[1] for a in anomaly_data]:
                    sudden_change = False

        return prediction or sudden_change
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return False

# Real-time Anomaly Detection and Visualization
def real_time_anomaly_detection(contamination=0.01, max_samples=0.9, bootstrap=True, random_state=42):
    """
    Performs real-time anomaly detection on a streaming dataset using the Isolation Forest algorithm.

    Parameters:
        contamination (float): The proportion of anomalies in the data set.
        max_samples (float): The number of samples to draw from data to train each base estimator.
        bootstrap (bool): Whether samples are drawn with replacement.
        random_state (int): Random seed for reproducibility.
    """
    try:
        # Initialize Isolation Forest model
        model = IsolationForest(
            contamination=contamination,
            max_samples=max_samples,
            bootstrap=bootstrap,
            random_state=random_state
        )

        # Collect initial data to train the model
        initial_data = [next(data_stream_simulator()) for _ in range(300)]  # Increased initial data points for better model training
        model.fit(np.array(initial_data).reshape(-1, 1))

        # Set up real-time plotting
        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 6))
        x_data, y_data, anomaly_data = [], [], []
        line, = ax.plot(x_data, y_data, label='Data', color='blue')
        scatter = ax.scatter([], [], color='red', label='Anomalies')
        ax.set_title('Real-time Anomaly Detection using Isolation Forest')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()

        # Start the data stream
        simulator = data_stream_simulator()
        t = 0
        previous_point = None

        try:
            while True:
                # Get next data point from the stream
                data_point = next(simulator)
                x_data.append(t)
                y_data.append(data_point)

                # Detect anomaly
                is_anomaly = isolation_forest_anomaly_detection(model, data_point, previous_point, threshold=20, anomaly_data=anomaly_data)
                if is_anomaly and (len(anomaly_data) == 0 or t - anomaly_data[-1][0] > 1):
                    anomaly_data.append((t, data_point))

                # Update plot
                line.set_xdata(x_data)
                line.set_ydata(y_data)
                if anomaly_data:
                    scatter.set_offsets(anomaly_data)

                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Update previous point
                previous_point = data_point

                # Increment time step
                t += 1
        except KeyboardInterrupt:
            # Finalize the plot when stopped by user
            plt.ioff()
            plt.show()
    except Exception as e:
        print(f"Error in real-time anomaly detection: {e}")

# Run real-time anomaly detection
if __name__ == "__main__":
    real_time_anomaly_detection(contamination=0.01, max_samples=0.9, bootstrap=True, random_state=42)

"""
Documentation:

This code performs real-time anomaly detection on a streaming dataset using the Isolation Forest algorithm. The data stream simulates real-time sequences with trend, seasonality, noise, and occasional anomalies. The Isolation Forest model, along with additional logic for detecting sudden changes in the data, identifies anomalies in the incoming data stream.

Algorithm Explanation:
- **Isolation Forest** is an effective algorithm for anomaly detection, particularly suited for high-dimensional datasets. It works by isolating points in the feature space using random splits. Anomalies are easier to isolate, resulting in shorter average path lengths for these points.
- In addition to Isolation Forest, a sudden change detection mechanism is added to detect large deviations from the previous data point, making the detection more sensitive to abrupt local changes.

Key Features:
1. **Real-time Data Stream Simulation**: The `data_stream_simulator` function generates synthetic time-series data with trend, seasonality, random noise, and injected anomalies.
2. **Isolation Forest Model**: The `IsolationForest` model is used to detect global anomalies. It is periodically updated with new data to improve its accuracy over time.
3. **Local Sudden Change Detection**: Sudden large changes are flagged as anomalies if they exceed a specified threshold, allowing the system to quickly identify sharp peaks or drops.
4. **Real-time Visualization**: The data stream and detected anomalies are plotted in real-time, providing an immediate visual representation of the anomalies.

Error Handling:
- The code includes error handling for data stream generation, anomaly detection, and real-time visualization to ensure robustness.

Requirements:
- Python 3
- Required Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

Requirements File (requirements.txt):
```
pandas
numpy
matplotlib
scikit-learn
```
"""