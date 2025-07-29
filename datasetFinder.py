#!/usr/bin/env python

import os
import zipfile

def find_datasets():
    """Find and extract datasets in common locations"""
    
    # Common download locations
    search_paths = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"), 
        ".",
        "C:/Users/Administrator/Downloads" if os.name == 'nt' else None
    ]
    
    # Remove None values
    search_paths = [p for p in search_paths if p]
    
    # Look for dataset files
    dataset_files = []
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if any(keyword in file.lower() for keyword in ['brain', 'tumor', 'mri', 'dataset', 'archive']):
                    if file.endswith('.zip'):
                        dataset_files.append(os.path.join(path, file))
    
    if not dataset_files:
        print("No dataset files found.")
        print("Please download dataset from Kaggle manually.")
        return None
    
    print(f"Found {len(dataset_files)} dataset file(s):")
    for i, file in enumerate(dataset_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    if len(dataset_files) == 1:
        selected_file = dataset_files[0]
    else:
        try:
            choice = int(input("Select file number: ")) - 1
            selected_file = dataset_files[choice]
        except:
            selected_file = dataset_files[0]
    
    # Extract the dataset
    extract_path = os.path.splitext(selected_file)[0]
    
    if not os.path.exists(extract_path):
        print(f"Extracting {os.path.basename(selected_file)}...")
        with zipfile.ZipFile(selected_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # Find the actual data directory
    data_dirs = []
    for root, dirs, files in os.walk(extract_path):
        # Look for directories with image files
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                   for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))):
                if root not in data_dirs:
                    data_dirs.append(root)
                break
    
    if data_dirs:
        data_dir = data_dirs[0]
    else:
        data_dir = extract_path
    
    print(f"Dataset ready at: {data_dir}")
    
    # Show structure
    if os.path.exists(data_dir):
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        if classes:
            print(f"Classes found: {classes}")
            for cls in classes:
                cls_path = os.path.join(data_dir, cls)
                count = len([f for f in os.listdir(cls_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {cls}: {count} images")
    
    return os.path.abspath(data_dir)

if __name__ == "__main__":
    data_path = find_datasets()
    if data_path:
        print(f"\nUse this path in your code:")
        print(f"data_dir = r'{data_path}'")