# Early-Breast-Cancer-Detection-using-Deep-Learning


# 🩺 Breast Cancer Detection with Deep Learning

This repository contains code to preprocess and analyze breast cancer CBIS-DDSM and histopathology images, train various deep learning models (CNN, EfficientNetB0, VGG16, MobileNet), and evaluate their performance. The goal is to compare the effectiveness of these models in classifying breast cancer images into benign and malignant categories, ultimately identifying the most effective model for breast cancer detection using histopathological images.

---

## 📝 Input and Output Formats

### 📥 Inputs
1. **Image Files**: Breast cancer CBIS-DDSM and histopathology images, categorized into benign and malignant classes.
2. **CSV Files**: Contain image paths and corresponding labels.
3. **Hyperparameters**: Parameters for model training such as learning rate, batch size, number of epochs, etc.

### 📤 Outputs
1. **Trained Models**: Saved models after training.
2. **Evaluation Metrics**: Accuracy, loss, precision, recall, F1 score, and ROC-AUC score.
3. **Confusion Matrices**: Visual representations of the model's performance on test data.
4. **Graphs**: Plots showing training and validation accuracy/loss over epochs.

---

## ⚙️ Function Descriptions

### 1. `load_data(file_path)`
**Purpose:** Loads image file paths and labels from a CSV file.  
**Parameters:**
- `file_path` (str): Path to the CSV file containing image paths and labels.  
**Returns:** A DataFrame with image paths and corresponding labels.

### 2. `preprocess_images(image_paths, labels, img_size=(50, 50))`
**Purpose:** Reads and resizes images, converts them to numpy arrays, and normalizes pixel values.  
**Parameters:**
- `image_paths` (list): List of image file paths.
- `labels` (list): List of labels corresponding to the images.
- `img_size` (tuple): Desired image size after resizing (default is `(50, 50)`).  
**Returns:** Numpy arrays of processed images and their corresponding labels.

### 3. `train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, learning_rate=0.001)`
**Purpose:** Trains the specified model on the training data and evaluates it on the validation data.  
**Parameters:**
- `model` (Keras Model): The deep learning model to be trained.
- `X_train` (numpy array): Training images.
- `y_train` (numpy array): Training labels.
- `X_val` (numpy array): Validation images.
- `y_val` (numpy array): Validation labels.
- `batch_size` (int): Number of samples per gradient update (default is `32`).
- `epochs` (int): Number of epochs to train the model (default is `50`).
- `learning_rate` (float): Learning rate for the optimizer (default is `0.001`).  
**Returns:** Trained model and history object containing training and validation metrics.

### 4. `evaluate_model(model, X_test, y_test)`
**Purpose:** Evaluates the trained model on test data and calculates various performance metrics.  
**Parameters:**
- `model` (Keras Model): The trained model to be evaluated.
- `X_test` (numpy array): Test images.
- `y_test` (numpy array): Test labels.  
**Returns:** Dictionary containing accuracy, loss, precision, recall, F1 score, confusion matrix, and ROC-AUC score.

### 5. `plot_training_history(history)`
**Purpose:** Plots the training and validation accuracy and loss over epochs.  
**Parameters:**
- `history` (History object): History object returned by the `fit` method of the model.  
**Returns:** None (displays the plot).

### 6. `plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)`
**Purpose:** Plots the confusion matrix.  
**Parameters:**
- `cm` (array): Confusion matrix to be plotted.
- `classes` (list): List of class names.
- `normalize` (bool): Whether to normalize the values (default is `False`).
- `title` (str): Title of the plot (default is 'Confusion matrix').
- `cmap` (Colormap): Colormap to be used (default is `plt.cm.Blues`).  
**Returns:** None (displays the plot).

### 7. `plot_roc_curve(fpr, tpr, roc_auc)`
**Purpose:** Plots the ROC curve.  
**Parameters:**
- `fpr` (array): False Positive Rate.
- `tpr` (array): True Positive Rate.
- `roc_auc` (float): Area Under the ROC Curve (AUC).  
**Returns:** None (displays the plot).

---

## 🏁 Conclusion

This code provides a robust pipeline for processing CBIS-DDSM and histopathological images, training and evaluating various deep learning models, and visualizing comprehensive performance metrics. The modular design of the functions allows for easy adaptation to different datasets and models, making this repository a valuable resource for ongoing research and development in breast cancer detection.

---

If you find this project helpful, please consider **starring** 🌟 the repository and **sharing** it with others. Contributions are welcome—let's work together to advance breast cancer detection and improve outcomes!
