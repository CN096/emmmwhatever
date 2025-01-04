import pandas as pd

# Load the existing CSV file
csv_file = "predictions_converted1.csv"  # 修改为你的文件路径
df = pd.read_csv(csv_file)

# Check current row count
current_row_count = len(df)
target_row_count = 4999

# Fill rows if the current count is less than 5000
if current_row_count < target_row_count:
    # Starting ID for filler rows
    start_id = df["ID"].max() + 1 if not df.empty else 1
    
    # Create filler rows
    filler_rows = pd.DataFrame([{
        "ID": start_id + i,
        "image_id": 99999,
        "class_id": 9,
        "x_min": 0,
        "y_min": 0,
        "width": 0,
        "height": 0
    } for i in range(target_row_count - current_row_count)])
    
    # Append filler rows to the DataFrame
    df = pd.concat([df, filler_rows], ignore_index=True)

# Save the updated CSV file
output_csv = "predictions_converted_filled3.csv"
df.to_csv(output_csv, index=False)

print(f"CSV file saved with padding as: {output_csv}")
