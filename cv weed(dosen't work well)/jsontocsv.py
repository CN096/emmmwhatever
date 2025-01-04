import json
import pandas as pd
import os

# Load the JSON file
with open("D:\\Users\\CN096\\computervision\\predictions13.json", "r") as f:
    data = json.load(f)

# Prepare the CSV data
csv_data = []
for idx, item in enumerate(data, start=1):
    # Extract fields
    file_name = item["file_name"]
    class_name = item["class"]
    bbox = item["bbox"]
    size = item["size"]

    # Compute required fields
    image_id = int(file_name.split(".")[0])  # Extract numbers from file_name
    class_id = 0 if class_name == "weed" else 1
    x_min, y_min = bbox[0], bbox[1]
    width, height = size[0], size[1]

    # Append row to csv data
    csv_data.append({
        "ID": idx,
        "image_id": image_id,
        "class_id": class_id,
        "x_min": x_min,
        "y_min": y_min,
        "width": width,
        "height": height
    })
    

# Convert to pandas DataFrame
df = pd.DataFrame(csv_data)

# Save to CSV file
df.to_csv("predictions_converted1.csv", index=False)

print("CSV file saved as predictions_converted.csv")
print("File saved at:", os.path.abspath("predictions_converted1.csv"))