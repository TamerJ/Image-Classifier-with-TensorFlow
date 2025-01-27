import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import tensorflow_hub as hub

def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict(image_path, model, top_k=5):
    img_array = process_image(image_path)
    predictions = model.predict(img_array)
    top_k_indices = predictions[0].argsort()[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = top_k_indices
    return top_k_probs, top_k_classes

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Predict flower class from image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes (default is 5)')
    parser.add_argument('--category_names', type=str, default=None, help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()
    
    # Load the model with custom objects for KerasLayer
    model = load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # Perform prediction
    probs, classes = predict(args.image_path, model, args.top_k)

    # Load category names if available
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        # Map the class indices to flower names
        class_labels = [class_names[str(cls)] for cls in classes]
    else:
        class_labels = [str(cls) for cls in classes]


    print(f"Top {args.top_k} Predictions:")
    for i in range(args.top_k):
        print(f"Class: {class_labels[i]}, Probability: {probs[i]:.4f}")

if __name__ == '__main__':
    main()
