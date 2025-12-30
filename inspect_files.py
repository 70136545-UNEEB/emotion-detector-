import os

def inspect_file(file_path):
    print(f"\n=== Inspecting {file_path} ===")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        if len(lines) > 0:
            print("\nFirst 3 lines:")
            for i, line in enumerate(lines[:3]):
                print(f"  {i+1}: {repr(line.strip())}")
            
            # Analyze format
            first_line = lines[0].strip()
            separator = None
            
            if '\t' in first_line:
                separator = 'tab'
                parts = first_line.split('\t')
            elif ';' in first_line:
                separator = 'semicolon'
                parts = first_line.split(';')
            elif ',' in first_line and first_line.count(',') == 1:
                separator = 'comma'
                parts = first_line.split(',')
            else:
                # Try space separation
                parts = first_line.split(' ', 1)
                if len(parts) == 2:
                    separator = 'space'
                else:
                    separator = 'unknown'
            
            print(f"Separator: {separator}")
            
            if separator != 'unknown' and len(parts) == 2:
                label, text = parts[0].strip(), parts[1].strip()
                print(f"Label: '{label}'")
                print(f"Text: '{text}'")
                
        # Check for empty lines
        empty_lines = sum(1 for line in lines if not line.strip())
        if empty_lines > 0:
            print(f"Empty lines: {empty_lines}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    print("Inspecting dataset files...")
    
    files_to_check = ['data/train.txt', 'data/test.txt', 'data/val.txt', 'data/emotions.csv']
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            inspect_file(file_path)
        else:
            print(f"\nFile not found: {file_path}")

if __name__ == "__main__":
    main()