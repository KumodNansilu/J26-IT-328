import sys
sys.path.insert(0, ".")
from ai_services.video_facial_emotion_behavior.emotion_recognition import predict as P
print("ModelNotAvailableError:", hasattr(P, "ModelNotAvailableError"))
from ai_services.video_facial_emotion_behavior.emotion_recognition.cnn_model import build_model
m = build_model()
print("build_model ok")
import torch
torch.save(m.state_dict(), r"%TEMP%\w.pt")
print("save ok")
