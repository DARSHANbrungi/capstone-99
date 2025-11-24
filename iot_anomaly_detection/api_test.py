import requests
import time
import json

# The URL of your running app.py server
# Note: Using port 5001 as defined in your latest app.py
API_URL = "http://localhost:5001/api/get_live_data"

print(f"Starting test client...")
print(f"Querying API at: {API_URL}")
print("Press Ctrl+C to stop.")

def query_api():
    try:
        # Send the GET request to the API
        response = requests.get(API_URL)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Print the data in a clean format
            print("--- New Prediction ---")
            print(f"  Anomaly: {data.get('is_anomaly')}")
            print(f"  Score:   {data.get('anomaly_score')}")
            print("  KPIs:")
            for key, value in data.get('kpi_data', {}).items():
                print(f"    {key}: {value}")
            print("\n")
            
        else:
            # Print an error if the request failed
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            print("\n")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print(f"Is app.py running at {API_URL}?")
        print("\n")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("\n")

if __name__ == "__main__":
    while True:
        query_api()
        # Wait for 5 seconds before the next request
        time.sleep(5)
