import os
import csv
import re

# Path to videos directory
videos_dir = './recordings/test'
output_csv = '../../docs/test_glossary.csv'

# Get all video files
video_files = []
for filename in os.listdir(videos_dir):
    if filename.endswith('.mp4'):
        video_files.append(filename)

# Sort files for consistent ordering
video_files.sort()

# Extract gloss from filename (remove number prefix and extension)
glossary_data = []
for filename in video_files:
    # Get relative path
    video_path = os.path.join(videos_dir, filename)
    
    # Extract gloss: remove number prefix (e.g., "1-", "10-") and extension (.mp4)
    # Pattern: starts with digits, followed by dash, then the gloss word, then .mp4
    match = re.match(r'^\d+-(.+)\.mp4$', filename)
    if match:
        gloss = match.group(1)
        glossary_data.append({
            'Video file': filename,
            'Gloss': gloss
        })

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Video file', 'Gloss'])
    writer.writeheader()
    writer.writerows(glossary_data)

print(f"Created glossary with {len(glossary_data)} entries")
print(f"Output saved to: {output_csv}")

# Show sample entries
print("\nSample entries:")
for i, entry in enumerate(glossary_data[:10]):
    print(f"  {entry['Video file']} -> {entry['Gloss']}")
