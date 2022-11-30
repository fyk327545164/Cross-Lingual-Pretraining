from datasets import load_dataset

valid_datasets = load_dataset("xquad", f"xquad.ar")
valid_column_names = valid_datasets.column_names
print(valid_column_names)