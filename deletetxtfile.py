import os

def delete_empty_txt_files(folder_path, recursive=False):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    if recursive:
        txt_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
    else:
        txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

    print(f"Found {len(txt_files)} .txt files in the folder.")

    for txt_file in txt_files:
        file_size = os.path.getsize(txt_file)

        if file_size == 0:
            print(f"Deleting empty file: {txt_file}")
            os.remove(txt_file)
        else:
            print(f"File is not empty: {txt_file}, Size: {file_size} bytes")

if __name__ == "__main__":
    folder_path = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\qa"
    delete_empty_txt_files(folder_path, recursive=True)