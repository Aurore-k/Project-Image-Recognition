from mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def detect_and_display_faces(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    detector = MTCNN()

    faces = detector.detect_faces(image_array)

    plt.imshow(image_array)
    ax = plt.gca()

    for face in faces:
        if face['confidence'] > 0.95: 
            x, y, width, height = face['box']
            plt.text(x, y - 10, f'Face ({face["confidence"]:.2f})', color='red', fontsize=10)
            rect = plt.Rectangle((x, y), width, height, fill=False, color='red')
            ax.add_patch(rect)

    plt.axis('off')
    plt.show()

image_path = "/Users/aurorekouakou/image_recognition/dataset/face/pexels-jeffreyreed-769749.jpg"
detect_and_display_faces(image_path)
