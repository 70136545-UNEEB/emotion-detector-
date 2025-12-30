import os
import pandas as pd

def detect_format(file_path):
    """Detect the format of the data file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        if '\t' in first_line:
            return 'tsv'
        elif ';' in first_line:
            return 'semicolon'
        elif ',' in first_line and first_line.count(',') == 1:
            return 'csv_simple'
        else:
            parts = first_line.split(' ', 1)
            if len(parts) == 2:
                return 'space'
            else:
                return 'unknown'
    except:
        return 'unknown'

def convert_to_standard_format(input_file, output_file, input_format=None):
    """Convert various formats to standard label\ttext format"""
    if input_format is None:
        input_format = detect_format(input_file)
    
    print(f"Converting {input_file} from {input_format} format...")
    
    converted_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                if input_format == 'tsv':
                    label, text = line.split('\t', 1)
                elif input_format == 'semicolon':
                    label, text = line.split(';', 1)
                elif input_format == 'csv_simple':
                    label, text = line.split(',', 1)
                elif input_format == 'space':
                    label, text = line.split(' ', 1)
                else:
                    # Try to guess format
                    if '\t' in line:
                        label, text = line.split('\t', 1)
                    elif ';' in line:
                        label, text = line.split(';', 1)
                    elif ',' in line:
                        label, text = line.split(',', 1)
                    else:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            label, text = parts
                        else:
                            print(f"Skipping line {line_num}: cannot parse - {line}")
                            continue
                
                # Clean and write
                label = label.strip()
                text = text.strip()
                f_out.write(f"{label}\t{text}\n")
                converted_lines += 1
                
            except Exception as e:
                print(f"Error converting line {line_num}: {e}")
                print(f"Line content: {repr(line)}")
                continue
    
    print(f"Converted {converted_lines} lines to {output_file}")

def main():
    """Convert all dataset files to standard format"""
    files_to_convert = ['train.txt', 'test.txt', 'val.txt']
    
    # Create backup directory
    os.makedirs('data/backup', exist_ok=True)
    
    for filename in files_to_convert:
        input_file = f'data/{filename}'
        backup_file = f'data/backup/{filename}.backup'
        output_file = f'data/{filename}'
        
        if os.path.exists(input_file):
            # Backup original file
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"Backed up {input_file} to {backup_file}")
            
            # Convert file
            convert_to_standard_format(input_file, output_file)
        else:
            print(f"File not found: {input_file}")

if __name__ == "__main__":
    main()