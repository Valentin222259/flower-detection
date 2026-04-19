import tensorflow as tf
import numpy as np
import cv2
import sys

# Incarca modelul salvat
model = tf.keras.models.load_model('flower_model.keras')
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Incarca imaginea
img_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
img_array = tf.expand_dims(img_resized, 0)

# Prezice
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

# Afiseaza rezultatul pe imagine
label = f"{predicted_class}: {confidence:.1f}%"
cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
print(f"Floare detectata: {predicted_class} ({confidence:.1f}%)")

cv2.imshow('Flower Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()