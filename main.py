from pathlib import Path
from random import choice
import requests
from PIL import Image
import numpy as np
import openpifpaf

"""
TEIL 1 - OpenPifPaf

Face Alignment with OpenPifPaf

Using of the triangle keypoints to measure the face angle


Test data is from the following sources:
-> wiki images

Dataset could be:
- IMDB Covers over time and genre

TEIL 2 - Hugging Face Emotion Classification

Interesting could be a investigation of the emotions in different face angles
"""

openpifpaf.show.Canvas.show = True
openpifpaf.show.Canvas.image_min_dpi = 200

path = Path(__file__).parent / "test_images"
# image_url = choice(list(path.glob("*.jpg")))
for image_url in list(path.glob("*.jpg")):
    image = Image.open(image_url)
    im = np.asarray(image)

    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')
    predictions, gt_anns, image_meta = predictor.pil_image(image)

    annotation_painter = openpifpaf.show.AnnotationPainter()
    with openpifpaf.show.image_canvas(im) as ax:
        annotation_painter.annotations(ax, predictions)


"""
TEIL 2
"""
import torch
from torchvision import transforms
import openpifpaf
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Load the Hugging Face model and its feature extractor
model_name = "trpakov/vit-face-expression"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


# Function to align faces using OpenPifPaf
def align_faces(image):
    openpifpaf_decoder = openpifpaf.decoder.factory_decode(openpifpaf.network.Factory().head_nets)
    preprocess = openpifpaf.transforms.Compose([
        openpifpaf.transforms.NormalizeAnnotations(),
        openpifpaf.transforms.CenterPadTight(16),
        openpifpaf.transforms.EVAL_TRANSFORM
    ])
    pil_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data = preprocess({'image': pil_transform(image)})
    data_loader = torch.utils.data.DataLoader([data], batch_size=1, pin_memory=True)

    for batch in data_loader:
        predictions = openpifpaf_decoder.forward(batch['image'])
        return predictions


# Function to classify emotion of a face
def classify_emotion(face):
    inputs = feature_extractor(face, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


# Align faces in the image
aligned_faces = align_faces(image)

# Classify emotions of each face
for face in aligned_faces:
    emotion = classify_emotion(face)
    print(f"Emotion: {emotion}")
