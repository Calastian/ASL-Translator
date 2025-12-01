import pandas as pd
import ast

input_csv = './turnIn/src/training_data/christian/val.csv'
output_csv = './turnIn/src/training_data/christian/padded_val.csv'
target_length = 266
reference_col = 'poseLandmarks_present'  # this to determine sequence length
pad_values = -999

#create a converter to string-list into a Num list
def parse_list(s):
    if pd.isna(s):
        return []
    fixed = str(s).replace(',]',']')
    try:
        return [float(x) for x in ast.literal_eval(fixed)]  # Use float for coordinate data
    except:
        return []

# pad each sequence to target_length
def pad_to_len(lst, wanted=target_length, pad_value=pad_values):
    if len(lst) == 0:  # Handle empty lists
        return [pad_value] * wanted
    return lst + [pad_value] * (wanted - len(lst))

df = pd.read_csv(input_csv)

# Find all columns that contain list data (sequences that need padding)
sequence_columns = []
non_sequence_columns = ['Unnamed: 0', 'Video file', 'Gloss']  # These are not sequences

for col in df.columns:
    if col in non_sequence_columns:
        continue
    # Check if column contains list-like data
    sample_val = df[col].iloc[0]
    if pd.isna(sample_val):
        continue
    try:
        # Try to parse as list
        if str(sample_val).startswith('[') and str(sample_val).endswith(']'):
            sequence_columns.append(col)
    except:
        continue

print(f"Found {len(sequence_columns)} sequence columns to pad:")
for i, col in enumerate(sequence_columns):
    print(f"  {i+1:3d}. {col}")

# Parse all sequence columns
for col in sequence_columns:
    print(f"Parsing {col}...")
    df[col] = df[col].apply(parse_list)

# Use reference column to determine original length and filter
df['orig_len'] = df[reference_col].apply(len)
original_rows = len(df)
df = df[df['orig_len'] <= target_length].copy()
filtered_rows = len(df)

print(f"Original rows: {original_rows}")
print(f"After filtering sequences > {target_length}: {filtered_rows} rows remain")
print(f"Filtered out: {original_rows - filtered_rows} rows")

#create padded list indicator 
# df['was_padded'] = (df['orig_len'] < target_length).astype(int)

# Apply padding to all sequence columns
for col in sequence_columns:
    print(f"Padding {col}...")
    df[col] = df[col].apply(lambda x: pad_to_len(x, target_length, pad_values))

# Getting pad column (using reference column)
df['padding'] = df['orig_len'].apply(lambda l: [0]*l + [1]*(target_length-l))

# Getting pad column (using reference column)
df['padding'] = df['orig_len'].apply(lambda l: [0]*l + [1]*(target_length-l))

#figure out how to reorder columns
gloss_idx = df.columns.get_loc('Gloss')
# new_cols = ['was_padded', 'padding']
new_cols = ['padding']

cols = (
    list(df.columns[:gloss_idx + 1]) +
    new_cols +
    [c for c in df.columns if c not in new_cols and c not in df.columns[:gloss_idx + 1]]
)
df = df[cols]

df = df.drop(columns = ["orig_len"])

df.to_csv(output_csv, index=False)

print(f"\nDataset saved to {output_csv}")
print(f"Final dataset: {len(df)} rows")
print(f"Padded columns: {len(sequence_columns)} sequence columns")

# Verify padding worked
sample_lengths = {}
for col in sequence_columns[:5]:  # Check first 5 columns
    lengths = df[col].apply(len).unique()
    sample_lengths[col] = lengths

print(f"\nVerification - sequence lengths after padding:")
for col, lengths in sample_lengths.items():
    print(f"  {col}: {lengths}")

# print("Before padding:")
# print(df[[reference_col, 'orig_len']].head())

# print("After padding:")
# print(df[[reference_col, 'orig_len', 'was_padded', 'padding']].head())