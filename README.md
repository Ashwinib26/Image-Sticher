
# 🧵 Image Stitcher

### A robust panorama creation tool that automatically stitches overlapping images using classical computer vision techniques.

---

## 📌 What the Project Does

- Automatically stitches two or more overlapping images into a single seamless panorama.
- Provides a simple, user-friendly web interface for uploading and viewing results.
- Combines feature detection, transformation estimation, and image blending.

---

## 🧠 Concepts & Algorithms Involved

1. **SIFT (Scale-Invariant Feature Transform)**  
   - Detects and describes keypoints in both images.

2. **RANSAC (Random Sample Consensus)**  
   - Filters incorrect matches and estimates the homography matrix.

3. **Harris Corner Detector**  
   - Refines image alignment using strong corner features.

4. **Homography Transformation & Warping**  
   - Warps one image to align with the other using estimated transformation.

5. **Image Blending & Cropping**  
   - Blends the aligned images and removes black borders for a clean output.

---

## 🔁 Flow of Execution

1. **Upload**  : User uploads two overlapping images.

2. **Feature Detection & Matching (SIFT)** : Extracts keypoints and matches between images.

3. **Outlier Removal (RANSAC)** : Removes incorrect matches and computes homography.

4. **Refinement (Harris Detector)** : Enhances precision in aligning corners.

5. **Warping & Stitching** : Warps, blends, and crops final panorama for display.

6. **Display** : Stitched output is displayed directly on the same page.

---

## 🛠 Tech Stack

- **Frontend:** HTML + CSS (minimal, responsive upload/display page)
- **Backend:** Python (Flask)
- **Libraries:** OpenCV, NumPy, Matplotlib

---

## 📁 Additional Details

- 📓 A Jupyter Notebook is included in the repository for a detailed walkthrough of each algorithm and step.
- 🔗 For suggestions or issues, feel free to open a discussion in the Issues section or contact me via email at ashwinisbisen@gmail.com.

     If you like the project, give it a ⭐️, pull the repo, and give it a try!

---

## THANK YOU !!
