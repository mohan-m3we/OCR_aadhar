import cv2
import pytesseract
import re
import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Set up the path to Tesseract executable (adjust the path as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Change this path based on your system

# Helper Functions

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for better OCR results
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to clean up the image for better text extraction
    _, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(thresholded_image)

    return extracted_text

def extract_aadhar_details(text):
    """Extract Aadhar details like number and name from the OCR text."""
    aadhar_data = {}
    aadhar_pattern = r"\d{4}\s\d{4}\s\d{4}"  # Aadhar number pattern
    name_pattern = r"(?<=Name:)(.*?)(?=\n)"  # Example name pattern

    # Extract Aadhar number and name using regex
    aadhar_number = re.findall(aadhar_pattern, text)
    if aadhar_number:
        aadhar_data['Aadhar Number'] = aadhar_number[0]

    name = re.findall(name_pattern, text)
    if name:
        aadhar_data['Name'] = name[0].strip()

    return aadhar_data

def read_jsonl_file(file_path):
    """Read a JSONL file and return the data."""
    data = []
    with open('C:/Users/Lenovo/Desktop/OCR/_annotations.train.jsonl', 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def display_data(aadhar_data):
    """Display extracted Aadhar data in a nice format."""
    print("\n--- Extracted Aadhar Details ---")
    print(f"Aadhar Number: {aadhar_data.get('Aadhar Number', 'N/A')}")
    print(f"Name: {aadhar_data.get('Name', 'N/A')}")
    print("----------------------------------")

def process_document(image_path):
    """Process the document, extract text, and display Aadhar details."""
    try:
        # Step 1: Extract text from the image
        extracted_text = extract_text_from_image(image_path)

        if extracted_text:
            # Step 2: Extract Aadhar details from the text
            aadhar_details = extract_aadhar_details(extracted_text)
            display_data(aadhar_details)
        else:
            print("No text found in the image.")
    except Exception as e:
        print(f"Error: {e}")

def process_multiple_documents(image_paths):
    """Process multiple documents and display their details."""
    for image_path in image_paths:
        print(f"\nProcessing document: {image_path}")
        process_document(image_path)

# Handwritten Text Recognition (Optional AI Model)
def train_handwritten_model(dataset):
    """Train a simple AI model to recognize handwritten text patterns (dummy function)."""
    # This can be expanded with actual model training using deep learning.
    print("Training model on handwritten text patterns...")

def preprocess_handwritten_image(image_path):
    """Preprocess handwritten images for better OCR."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    return thresholded_image

def display_image(image):
    """Display image using matplotlib."""
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Read dataset (jsonl file) for analysis or training
    dataset_path = "path_to_your_dataset.jsonl"
    dataset = read_jsonl_file('C:/Users/Lenovo/Desktop/OCR/_annotations.train.jsonl')
    print(f"Loaded {len(dataset)} records from the dataset.")

    # Ask the user for the image input
    image_input = input("Enter the image path to process: ")

    # Process the provided image
    process_document(image_input)

    # Optionally, process multiple images
    # image_paths = ["path_to_image1.jpg", "path_to_image2.jpg"]
    # process_multiple_documents(image_paths)
