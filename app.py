from flask import Flask, request, render_template, send_file,url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# Set upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Add to Flask config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image1 = request.files['image1']
    image2 = request.files['image2']

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image1.jpg')
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'image2.jpg')
    image1.save(path1)
    image2.save(path2)

    stitched_path = stitch_images(path1, path2)

    if stitched_path:
        stitched_url = url_for('static', filename='results/stitched_result.jpg')
        return render_template('index.html', stitched_url=stitched_url)
    else:
        return render_template('index.html', stitched_url=None)

def stitch_images(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Feature detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match features
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image1 to image2's perspective
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Get size of final panorama
    corners_img1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    H_translate = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # Warp image1
    stitched_img = cv2.warpPerspective(img1, H_translate.dot(H), (xmax - xmin, ymax - ymin))
    
    # Paste image2 onto canvas
    stitched_img[translation[1]:height2 + translation[1], translation[0]:width2 + translation[0]] = img2

    result_path = os.path.join(RESULT_FOLDER, "stitched_result.jpg")
    cv2.imwrite(result_path, stitched_img)
    return result_path

if __name__ == "__main__":
    app.run(debug=True)
