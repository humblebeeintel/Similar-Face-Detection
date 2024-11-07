import os
import cv2
import re
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored
from tqdm import tqdm


# Define paths and constants
ORIGINAL_IMAGES_PATH = "original_images"
UNIQUE_FACES_PATH = "unique_faces"
SIMILARITY_THRESHOLD = 0.6  # Adjusted threshold for matching accuracy
EMBEDDING_MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"  #for face detection

def create_directories():
    os.makedirs(ORIGINAL_IMAGES_PATH, exist_ok=True)
    os.makedirs(UNIQUE_FACES_PATH, exist_ok=True)

def get_face_embedding(image_path):
    """
    Detects faces and generates embeddings using retinaface for detection
    and alignment and FaceNet512 for embedding extraction.
    """
    try:
        embeddings = DeepFace.represent(
            img_path=image_path,
            model_name=EMBEDDING_MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        return [embedding["embedding"] for embedding in embeddings] if embeddings else []
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def load_existing_embeddings():
    """Loads all stored embeddings and unique face folders."""
    embeddings_data = []
    for unique_face_folder in os.listdir(UNIQUE_FACES_PATH):
        embedding_path = os.path.join(UNIQUE_FACES_PATH, unique_face_folder, "embedding.npy")
        if os.path.exists(embedding_path):
            stored_embedding = np.load(embedding_path)
            embeddings_data.append((stored_embedding, unique_face_folder))
    return embeddings_data

def save_unique_face(face_embedding, face_image, unique_face_id):
    """Saves a unique face's embedding and cropped image to disk."""
    unique_face_folder = os.path.join(UNIQUE_FACES_PATH, f"face_{unique_face_id}")
    os.makedirs(unique_face_folder, exist_ok=True)

    # Save face image and embedding
    face_image_path = os.path.join(unique_face_folder, f"face_{unique_face_id}.jpg")
    cv2.imwrite(face_image_path, face_image)

    embedding_path = os.path.join(unique_face_folder, "embedding.npy")
    np.save(embedding_path, np.array(face_embedding))

    print(f"New unique face added to database in folder: {unique_face_folder}")

def process_image(image_path, embeddings_data):
    """Processes a single image, detects faces, and saves unique faces."""
    image_name = os.path.basename(image_path)
    original_image_dest = os.path.join(ORIGINAL_IMAGES_PATH, image_name)
    if not os.path.exists(original_image_dest):
        cv2.imwrite(original_image_dest, cv2.imread(image_path))

    # Generate embeddings for all detected faces
    face_embeddings = get_face_embedding(image_path)
    
    for face_embedding in face_embeddings:
        match_found = False
        for stored_embedding, unique_face_folder in embeddings_data:
            similarity = cosine_similarity([face_embedding], [stored_embedding])[0][0]
            if similarity >= SIMILARITY_THRESHOLD:
                similarity_percentage = similarity * 100
                print(colored(f"Matching unique face found in folder: {unique_face_folder}, "
                              f"Similarity: {similarity_percentage:.2f}%", 'green'))
                match_found = True
                break

        if not match_found:
            unique_face_id = len(embeddings_data) + 1
            save_unique_face(face_embedding, cv2.imread(image_path), unique_face_id)
            embeddings_data.append((face_embedding, f"face_{unique_face_id}"))

def benchmark_model(test_folder, embeddings_data):
    """Benchmarks the model by testing top-1 accuracy on a test folder and outputs a binary confusion matrix
    along with similarity and dissimilarity scores."""
    TP, TN, FP, FN = 0, 0, 0, 0
    similarity_scores = {"TP": [], "FP": [], "FN": [], "TN": []}
    total_tests = 0

    ground_truth_labels = []
    predicted_labels = []

    # Get list of all test images for progress tracking
    test_images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Using tqdm to wrap the image processing loop for a single progress bar
    with tqdm(total=len(test_images), desc="Benchmarking images") as pbar:
        for filename in test_images:
            image_path = os.path.join(test_folder, filename)
            face_embeddings = get_face_embedding(image_path)
            match_found = False
            ground_truth_id = int(re.search(r'_(\d+)\.', filename).group(1))  # Extract ground truth ID

            for face_embedding in face_embeddings:
                similarities = [(cosine_similarity([face_embedding], [stored_embedding])[0][0], unique_face_folder) 
                                for stored_embedding, unique_face_folder in embeddings_data]
                if similarities:
                    top_similarity, top_match_folder = max(similarities, key=lambda x: x[0])

                    if top_similarity >= SIMILARITY_THRESHOLD:
                        top_match_id = int(top_match_folder.split('_')[1])
                        match_found = True
                        
                        # Track for confusion matrix and similarity scores
                        if ground_truth_id == top_match_id:
                            TP += 1  # True Positive: Correct match found
                            similarity_scores["TP"].append(top_similarity)
                            ground_truth_labels.append(1)  # Positive class
                            predicted_labels.append(1)  # Positive prediction
                        else:
                            FP += 1  # False Positive: Incorrect match found
                            similarity_scores["FP"].append(top_similarity)
                            ground_truth_labels.append(1)
                            predicted_labels.append(0)
                        break  # Stop checking other embeddings for this image

            if not match_found:
                FN += 1  # False Negative: No match found
                similarity_scores["FN"].append(0)  # No similarity since no match was found
                ground_truth_labels.append(1)
                predicted_labels.append(0)

            total_tests += 1
            pbar.update(1)  # Update the progress bar for each processed image

    # Calculate TN based on total possible negatives
    TN = total_tests - (TP + FP + FN)
    similarity_scores["TN"].extend([0] * TN)  # Assume TN have zero similarity

    # Calculate accuracy
    accuracy = (TP + TN) / total_tests * 100 if total_tests > 0 else 0
    print(f"Benchmark Accuracy: {accuracy:.2f}% for {total_tests} tests")

    # Construct and visualize confusion matrix
    conf_matrix = [[TP, FP], [FN, TN]]
    xlabels = ["Positive", "Negative"]
    ylabels = ["True", "False"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=xlabels, yticklabels=ylabels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Save the confusion matrix as an image
    plt.savefig("confusion_matrix_binary.png", bbox_inches='tight')
    plt.close()

    # Print average similarity scores
    print("\nAverage Similarity Scores:")
    for score_type, scores in similarity_scores.items():
        if scores:  # Avoid division by zero
            average_score = sum(scores) / len(scores)
            print(f"{score_type}: {average_score:.2f}")
        else:
            print(f"{score_type}: No matches")


def process_images_in_folder(folder_path, embeddings_data):
    """Processes all images in a specified folder in numerical order."""
    filenames = sorted(
        [filename for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    for filename in filenames:
        image_path = os.path.join(folder_path, filename)
        print(f"Processing image: {image_path}")
        process_image(image_path, embeddings_data)

def main():
    create_directories()
    embeddings_data = load_existing_embeddings()

    # Process images to build database
    path_to_images_folder = "test_emb2"
    #process_images_in_folder(path_to_images_folder, embeddings_data)

    # Benchmark the model
    path_to_test_folder = "test"
    # Uncomment the line below to run benchmarking
    benchmark_model(path_to_test_folder, embeddings_data)

# Run the main function
if __name__ == "__main__":
    main()
