from shutil import copyfile
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image


def split():

    directory_path = "datasets"
    train_val = 0.8

    # Get the list of all items in the directory
    items = os.listdir(directory_path)

    # Iterate through subdirectories
    for item in items:

        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            print(f"Files in subdirectory '{item}':")

            # Get the list of files in the subdirectory
            subdirectory_files = os.listdir(item_path)
            len_of_subdir = len(subdirectory_files)
            
            print(len_of_subdir, "is the length of files.")
            total_train = int(len_of_subdir * train_val)

            os.makedirs("split/training", exist_ok=True)
            os.makedirs("split/validation", exist_ok=True)

            for each_rows in subdirectory_files[0: total_train]:
                os.makedirs(f"split/training/{item}", exist_ok=True)
                copyfile(directory_path + "/" + item + "/" + each_rows, "split/training" + "/" + item + "/" + each_rows)

            for each_rows in subdirectory_files[total_train:]:
                os.makedirs(f"split/validation/{item}", exist_ok=True)
                copyfile(directory_path + "/" + item + "/" + each_rows, "split/validation" + "/" + item + "/" + each_rows)



def augment():

    directory_path = "split"
    items = os.listdir(directory_path)

    # Iterate through subdirectories
    for item in items:

        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            print(f"Files in subdirectory '{item}':")
            # Get the list of files in the subdirectory
            subdirectory_files = os.listdir(item_path)
            for each in subdirectory_files:
                print(each, "IS THE EACH")
                
                # Create output directory for augmented images
                output_dir = os.path.join("augmented_images", item, each)
                os.makedirs(output_dir, exist_ok=True)

                # Define the ImageDataGenerator
                datagen = ImageDataGenerator(
                    rotation_range = 45,
                    width_shift_range=0.25,
                    height_shift_range=0.25,
                    shear_range=0.25,
                    zoom_range=0.25,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    rescale= 1.0 / 225.0
                )


                # Loop over each image and augment it 8 times
                for each_rows in os.listdir(os.path.join(directory_path, item, each)):
                    img_path = os.path.join(directory_path, item, each, each_rows)
                
                    img = image.load_img(img_path, target_size=(224, 224))
                    numpy_image = image.img_to_array(img)
                    numpy_image = numpy_image.reshape((1,) + numpy_image.shape)

                    # Generate augmented images and save them to the output directory
                    for i, batch in enumerate(datagen.flow(numpy_image, batch_size=1, save_to_dir=output_dir, save_prefix=f'{each_rows[:-4]}_aug', save_format='jpeg')):
                        if i >= 7:  # Generate 8 augmented images per input image
                            break
split()
augment()