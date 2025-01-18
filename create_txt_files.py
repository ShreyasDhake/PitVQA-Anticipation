
import os

# path to images folder
images_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\images"
# path to QA folder
qa_folder = r"C:\Shreyas\Research Project\PitVQA_Anticipation\Test\qa"
# create QA folder if needed
if not os.path.exists(qa_folder):
    os.makedirs(qa_folder)

# go through images folder
for folder in os.listdir(images_folder):
    folder_path = os.path.join(images_folder, folder)  # images/video_01

    if os.path.isdir(folder_path):
        # create counterpart sub-folder in QA folder
        qa_subfolder_path = os.path.join(qa_folder, folder)  # QA/video_01
        if not os.path.exists(qa_subfolder_path):
            os.makedirs(qa_subfolder_path)

        # create txt file for each image
        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                txt_filename = file.replace('.png', '.txt')
                txt_filepath = os.path.join(qa_subfolder_path, txt_filename)
                open(txt_filepath, 'w').close()
