import os 
parent_folder = r"I:\blabla\dataset"

folders = [f for f in os.listdir(parent_folder) 
           if os.path.isdir(os.path.join(parent_folder, f))]

folders.sort()
for folder in os.listdir(parent_folder):
    path = os.path.join(parent_folder, folder)
    if os.path.isdir(path):
        print(folder, len(os.listdir(path)))

print(folders)