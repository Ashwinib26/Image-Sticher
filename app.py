from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_images():
    files = request.files.getlist("images")
    image_paths = []

    # Save uploaded images
    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        image_paths.append(path)

    # Process images (stitching logic)
    result_path = stitch_images(image_paths)
    
    return result_path if result_path else "Error in stitching images."

def stitch_images(image_paths):
    images = [cv2.imread(img) for img in image_paths]

    # Step 1: Feature Detection (SIFT)
    sift = cv2.SIFT_create()
    keypoints, descriptors = [], []
    
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Step 2: Feature Matching (FLANN)
    matcher = cv2.FlannBasedMatcher()
    matches = [matcher.knnMatch(descriptors[i], descriptors[i+1], k=2) for i in range(len(images)-1)]
    
    # Step 3: RANSAC for Homography Estimation
    homographies = []
    for i, match in enumerate(matches):
        good_matches = [m for m, n in match if m.distance < 0.75 * n.distance]

        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i+1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies.append(H)

    # Step 4: Warping & Stitching
    panorama = images[0]
    for i in range(1, len(images)):
        panorama = cv2.warpPerspective(panorama, homographies[i-1], 
                                       (panorama.shape[1] + images[i].shape[1], panorama.shape[0]))
        panorama[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]

    # Save result
    result_path = os.path.join(RESULT_FOLDER, "stitched_panorama.jpg")
    cv2.imwrite(result_path, panorama)
    
    return result_path

if __name__ == "__main__":
    app.run(debug=True)
