# Face Recognition Model

This project is a face recognition system that detects unique faces within a dataset and evaluates its accuracy using a binary confusion matrix. The system employs the FaceNet512 model for extracting facial embeddings, which are then used to measure the similarity between faces based on cosine similarity. The model also benchmarks its performance with a test set, calculating metrics like true positives, true negatives, false positives, and false negatives, while generating a confusion matrix for a visual summary.

## Key Components

### 1. **Face Embeddings with FaceNet512**
   - **Model**: This project uses the `FaceNet512` model for facial embeddings, a high-accuracy deep learning model for facial recognition.
   - **Embedding Process**: Using `DeepFace`, the model generates a 512-dimensional vector (embedding) for each detected face. This embedding captures unique features of the face and serves as a compact representation for comparison.
   - **Face Detection**: The `retinaface` backend is used for detecting and aligning faces before embedding generation, improving accuracy and consistency.

### 2. **Similarity Calculation**
   - **Cosine Similarity**: This metric is used to compare embeddings and determine the similarity between two faces. Cosine similarity ranges from -1 to 1, where 1 indicates identical embeddings.
   - **Similarity Threshold**: A threshold (0.6) is set to classify whether a new face is "similar enough" to an existing face in the database. If the similarity score exceeds this threshold, the face is considered a match; otherwise, it's classified as a new face.

### 3. **Data Storage Structure**
   - **Original Images**: The original images are saved in the `original_images` folder.
   - **Unique Faces Database**: Each unique face detected is stored in a separate subfolder under `unique_faces`, with each folder containing the face image and its corresponding embedding.

### 4. **Benchmarking and Confusion Matrix**
   - **Test Evaluation**: The model benchmarks itself using images in a designated test folder, comparing each test image's embedding against the stored embeddings.
   - **Confusion Matrix**: A binary confusion matrix (2x2) is generated to evaluate the model's accuracy. It includes:
     - **True Positives (TP)**: Correctly matched faces.
     - **True Negatives (TN)**: Correctly identified non-matches.
     - **False Positives (FP)**: Incorrect matches where a face is wrongly matched to an existing face.
     - **False Negatives (FN)**: Incorrectly missed matches where a face should have matched an existing face but didn't.
   - **Visualization**: The confusion matrix is visualized as a heatmap for easy analysis and saved as `confusion_matrix_binary.png`.

### 5. **Progress Tracking**
   - The `tqdm` library is used to display a single progress bar for all images in the test folder, allowing for a clear and continuous update on benchmarking progress.

### 6. **Similarity Scores**
   - Average similarity scores for TP, FP, FN, and TN are calculated and printed. These scores provide additional insight into how similar or dissimilar the model finds each type of match or mismatch.
