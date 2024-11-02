import os

# Covert .JPG to jpg in folder

for file in os.listdir('data/mit_dataset'):
    if '.JPG' in file:
        os.rename('data/mit_dataset/' + file, 'data/mit_dataset/' + file.replace('.JPG', '.jpg'))
        
# Remove all image with _gif

for file in os.listdir('data/mit_dataset'):
    if '_gif' in file:
        os.remove('data/mit_dataset/' + file)