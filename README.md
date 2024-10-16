# Real-time Anomaly Detection using Isolation Forest

This project performs real-time anomaly detection on a streaming dataset using the Isolation Forest algorithm. The data stream simulates real-time sequences with trend, seasonality, noise, and occasional anomalies. The Isolation Forest model, along with additional logic for detecting sudden changes in the data, identifies anomalies in the incoming data stream.

## Overview

The system generates synthetic time-series data with a mix of regular patterns and noise to simulate real-world scenarios like financial transactions or system metrics. The real-time anomaly detection helps identify unusual data points effectively.

## Algorithm Explanation

- **Isolation Forest** is used for anomaly detection. It is particularly effective for high-dimensional datasets by isolating data points using random splits. Anomalies are isolated more quickly, resulting in shorter average path lengths.
- A **sudden change detection mechanism** is added to capture abrupt changes that may not be flagged by the Isolation Forest model.

## Key Features

1. **Real-time Data Stream Simulation**: Generates synthetic time-series data with trend, seasonality, random noise, and injected anomalies.
2. **Isolation Forest Anomaly Detection**: Uses the `IsolationForest` model to detect global anomalies, with periodic updates for improved accuracy.
3. **Local Sudden Change Detection**: Detects sharp peaks or drops that exceed a specified threshold, marking them as anomalies.
4. **Real-time Visualization**: Displays the data stream and detected anomalies in real-time for immediate insights.

## Error Handling

The code is designed with robust error handling for:

- Data stream generation
- Anomaly detection logic
- Real-time visualization

## Requirements

- Python 3
- Required Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

## Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```
3. Run the `real_time_anomaly_detection.py` script to start the real-time anomaly detection.

## Usage

The script starts by training the Isolation Forest model on initial synthetic data and then processes incoming data points from the simulated stream. Anomalies are detected in real-time and visualized accordingly.

To stop the process, simply interrupt the script (e.g., using `CTRL+C`).

## Requirements File (requirements.txt)

```
pandas
numpy
matplotlib
scikit-learn
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Isolation Forest: [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- Inspiration for real-time data streaming and visualization techniques.
