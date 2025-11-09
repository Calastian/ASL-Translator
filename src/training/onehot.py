import pandas as pd
import numpy as np

df = pd.read_csv("../data/small_val.csv")
key_df = pd.read_csv("../../docs/Key_ASL.csv")
all_words = key_df['words'].tolist()
print(f"Total words from key_asl.csv: {len(all_words)}")
one_hot_matrix = np.zeros((len(df), len(all_words)))
for i, gloss in enumerate(df['Gloss']):
    if gloss in all_words:
        word_index = all_words.index(gloss)
        one_hot_matrix[i, word_index] = 1
    else:
        base_gloss = ''.join(char for char in gloss if not char.isdigit())
        if base_gloss in all_words:
            word_index = all_words.index(base_gloss)
            one_hot_matrix[i, word_index] = 1
            print(f"Mapped '{gloss}' to '{base_gloss}'")
        else:
            print(f"Warning: '{gloss}' (and base '{base_gloss}') not found in key_asl.csv")

encoded_df = pd.DataFrame(one_hot_matrix, columns=all_words)
result_df = df[["Video file"]].join(encoded_df)

result_df.to_csv('../data/small_val_encoding.csv', index=False)
print(f"One-hot encoded data saved to training_encoding.csv")
print(f"Shape: {result_df.shape}")
print(f"Sample of encoded data:")
print(result_df.head())
