from PIL import Image

from pest_classifier.data import build_eval_transform, build_train_transform
from pest_classifier.model import build_model


def test_model_has_requested_number_of_classes():
    model = build_model(num_classes=3, pretrained=False)
    assert model.classifier[1].out_features == 3


def test_eval_transform_returns_image_tensor():
    image = Image.new("RGB", (64, 48), color=(120, 80, 40))
    tensor = build_eval_transform(224)(image)
    assert tuple(tensor.shape) == (3, 224, 224)


def test_train_transform_returns_image_tensor():
    image = Image.new("RGB", (256, 256), color=(120, 80, 40))
    tensor = build_train_transform(224)(image)
    assert tuple(tensor.shape) == (3, 224, 224)
