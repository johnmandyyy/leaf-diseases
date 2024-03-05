
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import datetime
import shutil
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from keras.callbacks import *
from keras.losses import *
from keras import optimizers
import seaborn as sns
import numpy as np
from PIL import Image
import cv2

class ImageAugmentation:

  folder_name = "augmented_dataset" # The default directory for generating images.
  accepted_files = ["jpg", "png"]
  train_directory = folder_name + "/" + "train"
  validation_directory = folder_name + "/" + "validation"
  
  def resize_image(self, img):
    target_size = (224, 224)
    img = img.resize(target_size)
    return img

  def __init__(self, angle: int, size: tuple, allotment: tuple):
    """
    Create an instance for sizes:
      angle = 45
      allotment = (0.8, 0.2) - 80 Percent Training 20 Percent Validation

    Instantiates a new folder named augmented_dataset
    """
    self.angle = angle
    self.size = size
    self.list_of_classes = []
    self.allotment = allotment
    self.intializeDirectory()

    if isinstance(allotment, tuple) == False:
      raise ValueError("Allotment must be a tuple.")

    if isinstance(size, tuple) == False:
      raise ValueError("Sizes must be a tuple.")

    if isinstance(angle, int) == False:
      raise ValueError("Angle must be an int.")

  def createSplit(self, training_path, class_path) -> None:
      """
      Create a splitting based on tuple values
        values: (0.8, 0.2) Train / Validation
        path: Path of the classified images.
      """
      subdirectory_path = os.path.join(training_path, class_path)
      files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
      accepted_files = [file for file in files if file.split('.')[-1].lower() in self.accepted_files]
      print(accepted_files)

      parent_directory = os.path.dirname(training_path)
      validation_path = os.path.join(parent_directory, 'validation')
      os.makedirs(validation_path, exist_ok=True)

      if len(accepted_files) > 0:
          os.makedirs(os.path.join(validation_path, class_path), exist_ok=True)
          training_files, validation_files = train_test_split(
              accepted_files,
              test_size=self.allotment[1],
              random_state = 42  # You can set a random state for reproducibility
          )

          if len(validation_files) == 0:
              raise ValueError("Data is insufficient, the required file for each class must be at least 5 or greater than.")

          # Move files to the 'validation' folder
          validation_images = []
          for file in validation_files:
              source_path = os.path.join(subdirectory_path, file)
              destination_path = os.path.join(validation_path, class_path, file)
              print("From source training source path:", source_path, "moving to validation path:", destination_path)
              validation_images.append(str(destination_path))
              shutil.move(source_path, destination_path)

          return validation_images

  def showSubdirectory(self, folder_path: str) -> list:
      """
      Show the subdirectories inside the specified folder.

      Parameters:
      - folder_path (str): The path to the folder to analyze.

      Returns:
      - List of subdirectories.
      """
      sub_dir = []
      try:
          # Get the list of subdirectories in the specified folder
          subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

          if subdirectories:
              print("Subdirectories in", folder_path + ":")
              for subdir in subdirectories:
                print(subdir)
                sub_dir.append(subdir)
          else:
              return []

          return sub_dir

      except FileNotFoundError:
          return []

  def checkDirectory(self, folder_name: str) -> bool:
      """
      Check if the specified folder exists.
      Parameters:
      - folder_name (str): The name of the folder to check .
      """
      if os.path.exists(folder_name):
        return True
      return False

  def intializeDirectory(self) -> None:
      """
      Creates a directory needed for augmented dataset.
      returns None
      """

      if not os.path.exists(self.folder_name):
          # Create the folder
          os.makedirs(self.folder_name)
          print(f"The folder '{self.folder_name}' has been created.")
      else:
          print(f"The folder '{self.folder_name}' already exists. Deleting and recreating.")

          # Delete the existing folder and its contents
          shutil.rmtree(self.folder_name)

          # Recreate the folder
          os.makedirs(self.folder_name)
          print(f"The folder '{self.folder_name}' has been recreated.")

      # Create folder for the validation.
      if not os.path.exists(self.validation_directory):
          # Create the folder
          os.makedirs(self.validation_directory)
          print(f"The folder '{self.validation_directory}' has been created.")
      else:
          print(f"The folder '{self.validation_directory}' already exists. Deleting and recreating.")

          # Delete the existing folder and its contents
          shutil.rmtree(self.validation_directory)

          # Recreate the folder
          os.makedirs(self.validation_directory)
          print(f"The folder '{self.validation_directory}' has been recreated.")

  def getClasses(self):
    return self.list_of_classes

  def generateImages(self, training_path: str) -> None:

      """
      Generate images
      based on angle and save to output.
      """

      has_images = False

      if isinstance(training_path, str) == False:
        raise ValueError("Path must be string")

      if isinstance(self.folder_name, str) == False:
        raise ValueError("Output must be string")

      if self.checkDirectory(training_path) == False:
        raise ValueError("Directory does not exist at all.")

      directory = self.showSubdirectory(training_path)

      if len(directory) < 1:
        raise ValueError("There is no subdirectory for the class itself, make sure there is a predicting image in that class")

      for each_directory in directory:
        print("Working on the", each_directory, "folder")

        subdirectory_path = os.path.join(training_path, each_directory)
        files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
        accepted_files = [file for file in files if file.split('.')[-1].lower() in self.accepted_files]

        if len(accepted_files) < 1:
          continue
        else: # For validation files.
          self.list_of_classes.append(
              each_directory
          ) # To save the class names.
          validation_images = self.createSplit(training_path, each_directory) # Splits the training file and validation file.

          for each_file in validation_images:

            # Manual Augmentation
            datagen = ImageDataGenerator(
                rotation_range = self.angle,
                width_shift_range= 0.1,
                height_shift_range=0.1,
                shear_range= 0.1,
                zoom_range= 0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            img = image.load_img(each_file, target_size=self.size)

            numpy_image = image.img_to_array(img)
            numpy_image = numpy_image.reshape((1,) + numpy_image.shape)
            rotation_counter = 0

            for batch in datagen.flow(numpy_image, batch_size=1):
                plt.figure(rotation_counter)
                imgplot = plt.imshow(image.array_to_img(batch[0]))
                plt.axis('off')  # Turn off axis
                rotation_counter += 1

                # Save the augmented image

                temporary_save_path = self.validation_directory + "/" + each_directory

                if os.path.exists(temporary_save_path):
                    print("Directory does exist.")
                else:
                    os.makedirs(temporary_save_path)

                current_datetime = datetime.datetime.now()
                timestamp_str = current_datetime.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(temporary_save_path, f"augmented_image_{timestamp_str}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

                if rotation_counter % 8 == 0:  # 8 Augmented Images
                    break


        # Count again the files.
        files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
        accepted_files = [file for file in files if file.split('.')[-1].lower() in self.accepted_files]

        if os.path.exists(self.train_directory + "/" + each_directory):
            print("Directory does exist.", self.train_directory + "/" + each_directory, "Deleteing the folder recursively.")
            shutil.rmtree(temporary_save_path)

        for each_file in accepted_files:

          f_path = subdirectory_path + "/" + each_file

          # Manual Augmentation
          datagen = ImageDataGenerator(
              rotation_range = self.angle,
              width_shift_range= 0.1,
              height_shift_range=0.1,
              shear_range= 0.1,
              zoom_range= 0.1,
              horizontal_flip=True,
              fill_mode='nearest'              
          )

          img_path = f_path
          img = image.load_img(img_path, target_size=self.size)

          numpy_image = image.img_to_array(img)
          numpy_image = numpy_image.reshape((1,) + numpy_image.shape)
          rotation_counter = 0

          for batch in datagen.flow(numpy_image, batch_size=1):
              plt.figure(rotation_counter)
              imgplot = plt.imshow(image.array_to_img(batch[0]))
              plt.axis('off')  # Turn off axis
              rotation_counter += 1

              # Save the augmented image

              temporary_save_path = self.train_directory + "/" + each_directory

              if os.path.exists(temporary_save_path):
                  print("Directory does exist.")
              else:
                  os.makedirs(temporary_save_path)

              current_datetime = datetime.datetime.now()
              timestamp_str = current_datetime.strftime("%Y%m%d_%H%M%S")
              save_path = os.path.join(temporary_save_path, f"augmented_image_{timestamp_str}.png")
              plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


              if rotation_counter % 8 == 0:  # 8 Augmented Images
                  break

          #plt.show()

class CNN:

  #Augmentor = ImageAugmentation(45, (224, 224), (0.8, 0.2))
  #Augmentor.generateImages("datasets")
  save_best = True

  def __init__(self):
    self.model = None

  def applicationVGG(self):

    vgg_sixteen = VGG16(include_top=False, weights='imagenet', input_shape=(224,224, 3)) # +3 IF PICTURE IS IN RGB +1 IF BLACK AND WHITE
    for layer in vgg_sixteen.layers[:-4]:
        layer.trainable = False

    x = Flatten()(vgg_sixteen.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Adding dropout for regularization
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)  # Adding dropout for regularization
    prediction = Dense(6, activation='softmax')(x)
    
    model = Model(inputs=vgg_sixteen.input, outputs=prediction)
    model.summary()

    #opi = optimizers.Adam()
    #optimizers.SGD(learning_rate=0.008, momentum=0.9)
    opi = optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opi, loss=categorical_crossentropy,
                    metrics=['accuracy'])
  

    image_actions = ImageDataGenerator(preprocessing_function = preprocess_input)

    batch_size = 16

    train_generator = image_actions.flow_from_directory(
      directory='augmented_images/training',
      target_size=(224,224),
      shuffle=True,
      batch_size=batch_size,
    )

    # Save Labels.

    # Save the class indices to a dictionary
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())

    # Save the class names to a file
    file_path = "class_names.txt"

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted existing {file_path}.")

    with open(file_path, "w") as file:
        for class_name in class_names:
            file.write(class_name + "\n")

    print(f"Class names saved to {file_path}")

    valid_generator = image_actions.flow_from_directory(
      directory='augmented_images/validation',
      target_size=(224,224),
      shuffle=True,
      batch_size=batch_size,
    )


    #self.indices = train_generator.class_indices

    if self.save_best == True:

      checkpoint = ModelCheckpoint("vgg_16_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
      early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
      history = model.fit_generator(steps_per_epoch=len(train_generator),generator=train_generator, validation_data=valid_generator, validation_steps=len(valid_generator),epochs=25,callbacks=[checkpoint,early])
      model.save('vgg_16_model_final.h5')

      # Save the training and validation accuracy/loss plots
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['val_accuracy'])
      plt.title('Model Accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Validation'], loc='upper left')
      plt.savefig('accuracy_plot.png')
      plt.show()

      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('Model Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Validation'], loc='upper left')
      plt.savefig('loss_plot.png')
      plt.show()

      # Evaluate the model on the validation set
      evaluation = model.evaluate(valid_generator)
      print("Test Accuracy:", evaluation[1])

      y_true = valid_generator.classes
      y_pred = model.predict(valid_generator)
      y_pred_classes = [max(enumerate(pred), key=lambda x: x[1])[0] for pred in y_pred]

      cm = confusion_matrix(y_true, y_pred_classes)

    # Plot and save confusion matrix as a PNG file
      plt.figure(figsize=(len(class_names), len(class_names)))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
      plt.xticks(fontsize=6)
      plt.yticks(fontsize=6)
      plt.title('Confusion Matrix')
      plt.xlabel('Predicted Labels')
      plt.ylabel('True Labels')
      plt.savefig('confusion_matrix.png')
      plt.show()
      

      accuracy = accuracy_score(y_true, y_pred_classes)
      precision = precision_score(y_true, y_pred_classes, average='weighted')
      recall = recall_score(y_true, y_pred_classes, average='weighted')
      f1 = f1_score(y_true, y_pred_classes, average='weighted')
      print(accuracy, precision, recall, f1)


  def predictVGG16(self, name):
   
    class_names = ["Leaf Blights", "Leaf Curl", "Leaf Rust", "Leaf Spot", "Powdery Mildew", "Shot Hole"]
    imageLog = name


    img = Image.open(str(imageLog))
    img = img.resize((224,224))
    img.save(str(imageLog))

    img = image.load_img(str(imageLog),target_size=(224,224))
    
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    prepo = img

    from keras.models import load_model
    saved_model = load_model("vgg_16_model.h5")
    
    list_of_percentage = []
    iterations = 0
    output = saved_model.predict(img)
    for x in output[0]:
      list_of_percentage.append(str(round((x * 100),2)) + ":" + class_names[iterations])
      print(round((x * 100),2))
      iterations = iterations + 1

    highestProb = np.argmax(output)
    img = image.load_img(str(imageLog),target_size=(224,224))
    print("VGG16: ", str(class_names[highestProb]) + ": " + str(max(output[0]) * 100))

    img_file = cv2.imread(str(imageLog))

    # Adjust the font size (e.g., change 0.8 to a smaller value)
    font_size = 0.5

    cv2.putText(img_file, f'Predicted: {class_names[highestProb]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)

    cv2.imshow('Image Predicting', img_file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def getModel(self):
    return self.model

  def testModel(self):
    return None

  def getStatistics(self):
    pass

C = CNN()
C.applicationVGG()
#C.predictVGG16("test_image_3.jpg")