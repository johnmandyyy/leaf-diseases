import tkinter as tk
from tkinter import PhotoImage
import cv2
from PIL import Image, ImageTk
import numpy as np

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
from keras.models import load_model
import tensorflow as tf
import imutils
from tkinter import filedialog

class GradCAM:

    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output.shape) == 4:
                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output],
        )

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)


class ResponsiveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.title("Responsive Tkinter App")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        self.configure(bg="gainsboro")

        self.camera_running = False
        self.cap = None
        self.create_widgets()

    def create_widgets(self):
        AppLabel = tk.Label(
            self,
            text="Rambutan Leaf\nDisease Classifier",
            font=("Arial", 18),
            bg="gainsboro",
        )
        UploadButton = tk.Button(
            self, text="Upload", font=("Arial", 14), bg="royal blue", fg="white",
            command=self.upload_file
        )
        self.camera_button = tk.Button(
            self,
            text="Camera",
            font=("Arial", 14),
            bg="royal blue",
            fg="white",
            command=self.toggle_camera,
        )

        # Placeholder image
        image_path = "rambutan.png"
        image = PhotoImage(file=image_path)
        self.photo_image = image

        self.image_label = tk.Label(self, image=image, bg="gainsboro")
        self.image_label.image = image

        AppLabel.grid(row=0, column=0, sticky="nsew", pady=10)
        self.image_label.grid(row=1, column=0, sticky="nsew", pady=10)
        UploadButton.grid(row=2, column=0, sticky="nsew", pady=10)
        self.camera_button.grid(row=3, column=0, sticky="nsew", pady=10)

        for i in range(4):
            self.grid_rowconfigure(i, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def upload_file(self):
        file_path = filedialog.askopenfilename(title="Select a File")
        print(file_path)
        image = cv2.imread(file_path)
        self.predictVGG16(image)
        

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_running = True
        self.update_camera_feed()
        self.camera_button.config(text="Stop Camera")

    def predictVGG16(self, np_frame):
        # Resize the input numpy array to (224, 224) if needed
        if np_frame.shape[:2] != (224, 224):
            np_frame = cv2.resize(np_frame, (224, 224))

        class_names = [
            "Leaf Blights",
            "Leaf Curl",
            "Leaf Rust",
            "Leaf Spot",
            "Powdery Mildew",
            "Shot Hole",
        ]

        list_of_percentage = []
        iterations = 0
        saved_model = load_model("vgg_16_model.h5")
        output = saved_model.predict(np.expand_dims(np_frame, axis=0))

        for x in output[0]:
            list_of_percentage.append(
                str(round((x * 100), 2)) + ":" + class_names[iterations]
            )
            print(round((x * 100), 2))
            iterations = iterations + 1

        highestProb = np.argmax(output)
        print(
            "VGG16: ", str(class_names[highestProb]) + ": " + str(max(output[0]) * 100)
        )

        cam = GradCAM(saved_model, highestProb)
        prep = preprocess_input(np.expand_dims(np_frame, axis=0))
        heatmap = cam.compute_heatmap(prep)
        heatmap = cv2.resize(heatmap, (224, 224))
        (heatmap, output) = cam.overlay_heatmap(heatmap, np_frame, alpha=0.5)
        

        font_size = 0.5

        # Show the image.
        cv2.putText(
            np_frame,
            f"Predicted: {class_names[highestProb]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Image Predicting", np_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        output = np.vstack([np_frame, heatmap, output])
        output = imutils.resize(output, height=700)
        cv2.imshow("Image Heatmap", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict_image(self, image: np, is_upload = False) -> None:
        resized_frame = cv2.resize(image, (224, 224))
        self.predictVGG16(resized_frame)
        self.update_image("rambutan.png")

    def stop_camera(self):
        self.predict_image(self.frame)
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
        self.camera_button.config(text="Camera")
        self.update_image("rambutan.png")

    def update_camera_feed(self):
        ret, self.frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (400, 300))
            photo_image = ImageTk.PhotoImage(image=Image.fromarray(resized_frame))

            self.photo_image = photo_image
            self.image_label.config(image=photo_image)
            self.image_label.image = photo_image

        if self.camera_running:
            self.after(10, self.update_camera_feed)

    def update_image(self, path):
        image = PhotoImage(file=path)
        self.photo_image = image
        self.image_label.config(image=image)
        self.image_label.image = image


if __name__ == "__main__":
    app = ResponsiveApp()
    app.mainloop()
