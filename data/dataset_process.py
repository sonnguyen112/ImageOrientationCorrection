import os

# Loop folders in folder

for folder in os.listdir('data/indoorMIT/indoorCVPR_09/Images'):
    # print(folder)
    for file in os.listdir('data/indoorMIT/indoorCVPR_09/Images/' + folder):
        # Move file to new folder
        os.rename('data/indoorMIT/indoorCVPR_09/Images/' + folder + '/' + file, 'data/mit_dataset/' + file)