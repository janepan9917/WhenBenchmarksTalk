from src import DATA_DIR
from datasets import load_from_disk, Dataset, DatasetDict

dataset_dict = load_from_disk(DATA_DIR+'/stack1k')
train_dataset = dataset_dict['train']

def extract_python_file(files):
    for file in files:
        if file["path"].endswith(".py"):
            return {"content": file["content"]}
    return {"content": None}

new_data = [extract_python_file(row["files"]) for row in train_dataset if extract_python_file(row["files"])["content"]]

# Convert the new data to a Dataset
new_dataset = Dataset.from_list(new_data)

# Save the new dataset to the specified directory
new_dataset_dict = DatasetDict({"train": new_dataset})
new_dataset_dict.save_to_disk(DATA_DIR+'/stack1kText')

# Optional: Iterate over the new dataset and print the content of the files
for row in new_dataset:
    print(row["content"])
    break