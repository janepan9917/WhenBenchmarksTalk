import os
import boto3
from smart_open import open
from datasets import load_dataset, Dataset
from datasets.dataset_dict import DatasetDict
from src import DATA_DIR

# Initialize a session with AWS credentials
session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
s3 = session.client("s3")

# Function to download contents from S3
def download_contents(files):
    for file in files:
        s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            file["content"] = fin.read().decode(file["src_encoding"])
    return {"files": files}

# Load the dataset
ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True)

# Filter the dataset for 'Python' language in 'gha_language' and take the first 1000 repositories
filtered_ds = ds.filter(lambda row: row.get('gha_language') == 'Python').take(1000)

# Convert the streaming dataset to a regular Dataset for further processing
filtered_ds = Dataset.from_list([row for row in filtered_ds])

# Map the download_contents function to the filtered dataset
filtered_ds = filtered_ds.map(lambda row: download_contents(row["files"]))

# Save the dataset to the specified directory
dataset_dict = DatasetDict({"train": filtered_ds})
dataset_dict.save_to_disk(DATA_DIR)

# Optional: Iterate over the dataset and print the content of the files
for row in filtered_ds:
    for file in row["files"]:
        print(file["content"])
    break  



# for row in train_dataset:
#     for file in row["files"]:
#         if '.py' in file["path"]:
#             first_py = file['content']
#             print(file["content"])
#             break
#     break