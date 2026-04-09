import numpy as np
from PIL import Image
import io
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Nenhum interpretador TFLite encontrado.")

# Pega o caminho de backend/ (onde a pasta assets está)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_CONFIGS = {
    'Elementos Arquitetônicos': {
        'path': os.path.join(BASE_DIR, 'assets', 'models', 'arquitetura_5elementos.tflite'),
        'labels': ['Church', 'Door', 'Fronton', 'Tower', 'Window'],
        'size': 96
    },
    'Casa Corrente': {
        'path': os.path.join(BASE_DIR, 'assets', 'models', 'casas_coloniais.tflite'),
        'labels': ["Casa de Meia Morada", "Casas de Porta e Janela", "Sobrado de Frente Estreita"],
        'size': 96
    },
    'Janelas Históricas': {
        'path': os.path.join(BASE_DIR, 'assets', 'models', 'janelas_heritage.tflite'),
        'labels': ["BalconyDoor-PortaSacada", "PanelledWindow-Almofada", "PlainWindow-Calha", "SashWindow-Guilhotina"],
        'size': 96
    }
}

def predict_image(image_bytes: bytes, category: str):
    config = MODEL_CONFIGS.get(category, MODEL_CONFIGS['Elementos Arquitetônicos'])

    if not os.path.exists(config['path']):
        raise FileNotFoundError(f"Modelo não encontrado em: {config['path']}")

    interpreter = tflite.Interpreter(model_path=config['path'])
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((config['size'], config['size']))
    img_array = np.array(img)

    # Normalização INT8 (Edge Impulse)
    input_data = (img_array - 128).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Dequantização
    output_raw = interpreter.get_tensor(output_details[0]['index'])[0]
    output_data = (output_raw.astype(np.float32) - (-128)) * 0.00390625
    
    best_index = np.argmax(output_data)
    return config['labels'][best_index], float(output_data[best_index])