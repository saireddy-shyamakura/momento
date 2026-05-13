import torch
import clip
from PIL import Image

# Set the device for model execution.
# Change to "cuda" when running on a GPU-enabled machine.
device = "cpu"

# Load the CLIP model and preprocessing pipeline for the selected model.
model, preprocess = clip.load("ViT-B/32", device=device)


def upload_images(image_path):
    """Load an image, preprocess it, encode it with CLIP, and normalize the image embedding."""
    with torch.inference_mode():
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device=device)
        image_features = model.encode_image(image=image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


def search_images(texts_list, image_features):
    """Tokenize text queries, encode them with CLIP, compute similarity to the image, and print the best match."""
    with torch.inference_mode():
        texts = clip.tokenize(texts_list).to(device=device)

        text_features = model.encode_text(text=texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between the image embedding and each text embedding.
        similarity = image_features @ text_features.T

        best_score = -float("inf")
        best_index = -1

        # Find the highest-scoring text label.
        for i, score in enumerate(similarity[0]):
            if score.item() > best_score:
                best_score = score.item()
                best_index = i

        print("Best match:", texts_list[best_index])
        print("Score:", best_score)


# Example usage: encode the sample image and compare it against a set of text labels.
image_features = upload_images("images/dog.jpg")
search_images(["dog", "car", "person running"], image_features)
