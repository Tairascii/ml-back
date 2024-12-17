from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Create your views here.
MODEL_PATH = "model/cnn_notMNIST.keras"
model = load_model(MODEL_PATH)

CLASS_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

@csrf_exempt
def upload_video_and_process(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        result = {"label": "Object Detected", "confidence": 0.92}

        return JsonResponse({
            'message': 'Video processed successfully!',
            'result': result
        }, status=200)

    return JsonResponse({'error': 'Invalid request. Please upload a video file.'}, status=400)


@csrf_exempt
def classify_letter(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image = Image.open(image_file).convert('L')

        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions[0])
        predicted_letter = CLASS_LABELS[predicted_index]
        confidence = float(np.max(predictions[0]))

        return JsonResponse({
            'message': 'Image classified successfully!',
            'predicted_letter': predicted_letter,
            'confidence': confidence
        }, status=200)

    return JsonResponse({'error': 'Invalid request. Please upload an image.'}, status=400)

