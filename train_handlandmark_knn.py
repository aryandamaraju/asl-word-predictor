# train_handlandmark_knn.py
# Scans C:\ASL_Project\asl_images\<LABEL>\*.jpg, extracts MediaPipe hand landmarks,
# trains a KNN classifier, and saves the model to C:\ASL_Project\hand_landmark_knn.joblib

import os
import cv2
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mediapipe as mp

# ----------------------------
# Config - edit if needed
# ----------------------------
IMAGES_ROOT = r"C:\ASL PROJECT ASL\Sign Language for Alphabets" # folder with subfolders for each label
OUT_MODEL = r"C:\ASL PROJECT ASL\hand_landmark_knn.joblib"
USE_PCA = False          # set True to enable PCA if you have many dims / want to regularize
PCA_COMPONENTS = 30      # used only if USE_PCA = True
K_NEIGHBORS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_DETECTION_CONF = 0.5
# ----------------------------

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True,
                                max_num_hands=1,
                                min_detection_confidence=MIN_DETECTION_CONF)

def extract_landmarks_from_image(img_bgr):
    """Return 63-dim vector (x,y,z) normalized: wrist as origin and scaled."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])  # normalized positions
    # Translate so wrist (landmark 0) is origin (2D translation)
    pts[:, :2] -= pts[0, :2]
    # Scale by max 2D norm to be scale-invariant
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 1e-6:
        pts[:, :2] /= scale
    return pts.flatten()

def build_dataset(root):
    X = []
    y = []
    labels = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not labels:
        raise ValueError(f"No label subfolders found in {root}. Create e.g. {root}\\A, {root}\\B, ...")
    print("Found labels:", labels)
    for lab in labels:
        folder = os.path.join(root, lab)
        for fname in tqdm(os.listdir(folder), desc=f"Processing {lab}"):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            lm = extract_landmarks_from_image(img)
            if lm is not None:
                X.append(lm)
                y.append(lab)
    X = np.array(X)
    y = np.array(y)
    return X, y, labels

def main():
    print("Building dataset from:", IMAGES_ROOT)
    X, y, labels = build_dataset(IMAGES_ROOT)
    print("Raw dataset size (samples x dims):", X.shape)
    if X.shape[0] == 0:
        print("No landmarks extracted. Check images and lighting; ensure hands are visible.")
        return

    # Optional PCA
    pca = None
    if USE_PCA:
        print(f"Applying PCA -> {PCA_COMPONENTS} components")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        X = pca.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc*100:.2f}%")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))

    # Save model and metadata
    to_save = {'pca': pca, 'knn': knn, 'labels': labels}
    joblib.dump(to_save, OUT_MODEL)
    print("Saved model to:", OUT_MODEL)

if __name__ == "__main__":
    main()
