import cv2
import os

def create_video_from_images(image_folder, output_video_path, fps=1):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg") or img.endswith(".png")]
    
    if not images:
        print("No images found in the folder.")
        return

    
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    
    for image_name in images:
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (width, height))  
            video.write(img)

    video.release()
    print(f" Video created successfully at: {output_video_path}")


if __name__ == "__main__":
    create_video_from_images("data/sample_images", "data/videos/fundus_video.mp4", fps=1)
