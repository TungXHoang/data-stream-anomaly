import numpy as np
import matplotlib.pyplot as plt
import time


def simulation(n, base_level, noise, seasonal_amp, seasonal_freq):
    # Generate time intervals from 0 to n-1
    time = np.arange(n)
    
    # Create a seasonal component using a sine wave (periodic pattern)
    seasonal_component = seasonal_amp * np.sin(seasonal_freq * time)

    # Add base level, seasonal component, and random noise to create the data stream
    data_stream = base_level + seasonal_component + np.random.normal(loc=0, scale=noise, size=n)


    # Introduce random anomalies in the data 
    for i in range(n):
      if np.random.rand() < 0.02:  
        data_stream[i] += np.random.uniform(-15, 15)  # Random anomaly between -15 and 15
    
    return data_stream


def z_score_anomaly_detection(data_stream, threshold, window_size):
    anomalies = []  # List to store anomaly indices
    
    # Iterate over the data, starting from the end of the initial window
    for i in range(window_size, len(data_stream)):
        # Select a window of data points for anomaly detection
        window = data_stream[i-window_size:i]
        
        # Calculate mean and standard deviation for the window
        mean = np.mean(window)
        std_dev = np.std(window)
        
        if std_dev == 0:
          std_dev = 1
        
        # Compute Z-score
        z_score = (data_stream[i] - mean) / std_dev
        
        # Check if the Z-score exceeds the threshold (anomaly detection condition)
        if np.abs(z_score) > threshold:
          anomalies.append(i) 
    
    return anomalies


def anomaly_detection(data_stream, threshold, window_size, duration):
    start_time = time.time()  # Store the start time to manage duration
    anomalies = z_score_anomaly_detection(data_stream, threshold, window_size)  # Detect anomalies
    logged_anomalies = set()  # Set to store and avoid duplicate logging of anomalies

    plt.figure(figsize=(12, 6))

    # Loop through the data stream in real-time, updating the plot
    for i in range(window_size, len(data_stream)):
        plt.clf()
        plt.plot(data_stream[:i], label="Data Stream", color='blue')  # Plot the current portion of data

        # Filter out the anomalies detected up to the current point in time
        current_anomalies = [idx for idx in anomalies if idx < i]
        if current_anomalies:
          plt.scatter(current_anomalies, data_stream[current_anomalies], 
            color='red', label="Detected Anomalies", s=100)  # Mark anomalies in red

        # Log anomaly in terminal
        for anomaly in current_anomalies:
          if anomaly not in logged_anomalies:
            print(f"Anomaly detected at time {anomaly}, value: {data_stream[anomaly]}")
            logged_anomalies.add(anomaly) 


        plt.title("Real-Time Anomaly Detection in Data Stream")
        plt.xlabel("Time")
        plt.ylabel("Data Value")

        plt.legend() 
        plt.xlim(0,len(data_stream))  
        plt.ylim(0, 100)  
        plt.pause(0.05) 

      
        if time.time() - start_time > duration:
          break

    plt.show()  # Display the final plot

data_stream = simulation(n=1000, base_level=30, noise=1.5, seasonal_amp=5, seasonal_freq=0.01)

# Perform real-time anomaly detection on the generated data
anomaly_detection(data_stream, 3,  50, 75)
