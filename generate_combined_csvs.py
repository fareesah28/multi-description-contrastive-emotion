import pandas as pd

# input paths
input_csv_1 = "/mnt/scratch/users/adhg808/MAFW/Labels/generated_captions.csv"
input_csv_2 = "/mnt/scratch/users/adhg808/MAFW/Labels/generated_captions_more.csv"

# output paths
output_human = "/mnt/scratch/users/adhg808/MAFW/Labels/data_human.csv"
output_llava = "/mnt/scratch/users/adhg808/MAFW/Labels/data_llava.csv"

# load input CSVs
df1 = pd.read_csv(input_csv_1)
df2 = pd.read_csv(input_csv_2)

# create data_human.csv

df1_human = df1.iloc[:, :-1]  # remove last column (llava_caption_5)
df1_human.columns = ["video", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]

df2_human = df2.copy()
df2_human.columns = ["video", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]

data_human = pd.concat([df1_human, df2_human], ignore_index=True)
data_human.to_csv(output_human, index=False)

# create data_llava.csv

df1_llava = df1.drop(columns=["human_caption"])
df1_llava.columns = ["video", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]

df2_llava = df2.copy()
df2_llava.columns = ["video", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]

data_llava = pd.concat([df1_llava, df2_llava], ignore_index=True)
data_llava.to_csv(output_llava, index=False)

print(" Created:\n- data_human.csv\n- data_llava.csv")
