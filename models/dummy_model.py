import random

def predict_image(file_path):
    return random.choice(["Real", "Fake"]), round(random.uniform(0.7, 0.99), 2)

def predict_video(file_path):
    return random.choice(["Real", "Fake"]), round(random.uniform(0.7, 0.99), 2)

def predict_audio(file_path):
    return random.choice(["Real", "Fake"]), round(random.uniform(0.7, 0.99), 2)