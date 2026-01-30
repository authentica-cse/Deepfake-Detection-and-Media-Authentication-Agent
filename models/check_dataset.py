import os

def count_images(path):
    # Count only image files (jpg, jpeg, png)
    return len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print("Number of REAL images:", count_images("dataset/image/real"))
print("Number of FAKE images:", count_images("dataset/image/fake"))


