import cv2
import os

def extract_frames_from_video(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"frame_{frame_count:04d}.png"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        print(f" Saved: {frame_filename}")
        frame_count += 1

    cap.release()
    print(f"\n Total {frame_count} frames extracted to: {output_folder}")


if __name__ == "__main__":
    extract_frames_from_video("data/videos/fundus_video.mp4", "data/frames")
