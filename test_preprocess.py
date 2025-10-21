from preprocess import load_and_preprocess_images

images, names = load_and_preprocess_images("data/frames")

print(f"Loaded {len(images)} frames.")
print(f"Image shape: {images[0].shape}")
