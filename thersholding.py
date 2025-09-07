import cv2
import numpy as np
import os
import argparse
import glob
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fire_smoke.log"),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser(description="Fire & Smoke Detection on folder images")
parser.add_argument("--folder", type=str, required=True, help="Path to the folder with images")
parser.add_argument("--show", action="store_true", help="Show combined images (optional)")
args = parser.parse_args()

folder_path = args.folder
show_images = args.show

if not os.path.exists(folder_path):
    logging.error("Folder does not exist!")
    exit()

image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
              glob.glob(os.path.join(folder_path, "*.jpeg")) + \
              glob.glob(os.path.join(folder_path, "*.png"))

if len(image_paths) == 0:
    logging.error("No images found in folder!")
    exit()

save_folder = os.path.join(folder_path, "combined_results")
os.makedirs(save_folder, exist_ok=True)

total_images = len(image_paths)
for idx, img_path in enumerate(image_paths, 1):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)

    if img is None:
        logging.warning(f"Cannot read {img_name}, skipping...")
        continue


    B, G, R = cv2.split(img)


    fire_mask = (R > 200) & (R > G) & (R > B)
    fire_mask = fire_mask.astype(np.uint8)

    smoke_mask = (abs(R - G) < 15) & (abs(R - B) < 15) & (abs(G - B) < 15) & (R > 150)
    smoke_mask = smoke_mask.astype(np.uint8)

    segmented = img.copy()
    segmented[fire_mask == 1] = [0, 0, 255]
    segmented[smoke_mask == 1] = [200, 200, 200]

    combined = np.hstack((img, segmented))

    cv2.putText(combined, "Fire ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(combined, "Smoke ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    save_path = os.path.join(save_folder, f"combined_{img_name}")
    cv2.imwrite(save_path, combined)
    logging.info(f"[{idx}/{total_images}] Saved {save_path}")

    if show_images:
        cv2.imshow("Original vs Segmentation", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
