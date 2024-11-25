import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import json

# Load the model
model = load_model("Dermno_RenseNet_02.keras")
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
# Load validation dataset
val = val_datagen.flow_from_directory(
    directory=r"C:\Users\AVIGHYAT\dermno_copy\val",
    target_size=(256, 256),
    batch_size=32,
    seed=42,
    shuffle=False  # Important for consistency in labels
)

# Load class labels (no inversion needed)
with open('class_indices.json', 'r') as f:
    class_labels = json.load(f)

# Get ground truth labels
true_labels = val.classes

# Get predictions
predictions = model.predict(val, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Classification Report
print("\nClassification Report:")
report = classification_report(true_labels, predicted_classes, target_names=[class_labels[str(i)] for i in range(len(class_labels))])
print(report)

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[class_labels[str(i)] for i in range(len(class_labels))])

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
# Rotate and align labels for readability
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)  # Adjust y-axis label font size

plt.tight_layout()  # Ensure everything fits without overlap

# Save the confusion matrix as an image
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')  # High resolution and trimmed

plt.show()
