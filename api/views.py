from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import onnxruntime as ort
import io
from torchvision import transforms
import numpy as np
import cv2
import torch
from PIL import Image
import os

# Create your views here.
MODEL_PATH = "model/best.pt"
# onnx_session = ort.InferenceSession(MODEL_PATH)

checkpoint = torch.load(MODEL_PATH)
model = checkpoint['model']
model = model.float()

def preprocess_image(file):
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image

@csrf_exempt
def upload_image_and_process(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        pil_image = Image.open(image)
        image_np = np.array(pil_image.convert("RGB"))
        print(model)
        image_resized = cv2.resize(image_np, (640, 640))
        image_tensor = transforms.ToTensor()(image_resized)
        image_tensor = image_tensor.unsqueeze(0).float()

        with torch.no_grad():
            output = model(image_tensor)

        annotated_image = output[0].cpu().numpy()
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode('.png', annotated_image_bgr)
        current_directory = os.path.dirname(os.path.abspath(__file__))

        output_filename = "annotated_image.png"

        output_path = os.path.join(current_directory, output_filename)

        cv2.imwrite(output_path, annotated_image_bgr)
        response = HttpResponse(encoded_image.tobytes(), content_type="image/png")
        return response

    return JsonResponse({'error': 'Invalid request. Please upload a image file.'}, status=400)

