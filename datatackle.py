import csv
import os
import sys

def process_csv(input_file, metadata_file, values_file):
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return False
        
        # Open all files
        with open(input_file, 'r', encoding='utf-8-sig') as in_file, \
             open(metadata_file, 'w', newline='', encoding='utf-8') as meta_file, \
             open(values_file, 'w', newline='', encoding='utf-8') as val_file:
            
            # Setup CSV readers and writers
            reader = csv.DictReader(in_file)
            print("CSV列名:", reader.fieldnames)  # 添加调试信息
            
            meta_writer = csv.writer(meta_file)
            val_writer = csv.writer(val_file, quoting=csv.QUOTE_ALL)
            
            # Write headers
            meta_writer.writerow(['M4id', 'category', 'Frequency', 'Horizon', 'SP', 'StartingDate'])
            val_writer.writerow(['V1', 'V2'])
            
            # Process each row
            for idx, row in enumerate(reader, 1):
                try:
                    print(f"处理第{idx}行:", dict(row))  # 添加调试信息
                    # 处理可能带有BOM的列名
                    date_key = 'date'
                    if '\ufeffdate' in row:
                        date_key = '\ufeffdate'
                    
                    date = row[date_key]
                    close = row['rate']
                    m4id = f'D{idx}'
                    
                    # Write to metadata file
                    meta_writer.writerow([m4id, 'Finance', 1, 14, 'Daily', date])
                    
                    # Write to values file
                    val_writer.writerow([m4id, close])
                    
                except KeyError as e:
                    print(f"Error: Missing column {e} in input file.")
                    print(f"可用的列名: {list(row.keys())}")  # 添加调试信息
                    return False
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue
        
        print(f"Successfully created {metadata_file} and {values_file}")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example usage
if __name__ == "__main__":
    process_csv('英镑兑人民币_20250324_102930.csv', 'M4-info.csv', 'Daily-train.csv')