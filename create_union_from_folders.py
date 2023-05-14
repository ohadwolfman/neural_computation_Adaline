import os

# Specify the directory path containing the files
directory1 = 'קבצים (מ,ב,ל)'
directory2 = 'קבצים בהטייה של 15_ לשני הצדדים (מ,ב,ל)'

# Output file path

output_file = 'Data_15%_rotated_writing.txt'

# Initialize the merged content
merged_content = ''

# Iterate over each file in the directory
for filename in os.listdir(directory2):
    file_path = os.path.join(directory2, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read().strip()
            merged_content += content + '\n'

with open(output_file, 'w', encoding='utf-8') as file:
    file.write(merged_content)

print("Merged content has been written to the output file.")