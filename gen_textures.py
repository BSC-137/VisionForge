import numpy as np
from PIL import Image

def create_albedo(path, size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Checkered pattern
    for y in range(size):
        for x in range(size):
            r = 255 if (x // 32 + y // 32) % 2 == 0 else 50
            g = 100 if (x // 32 + y // 32) % 2 == 0 else 50
            b = 100 if (x // 32 + y // 32) % 2 == 0 else 255
            img[y, x] = [r, g, b]
    Image.fromarray(img).save(path)

def create_normal(path, size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # create some bumps
    for y in range(size):
        for x in range(size):
            nx = np.sin(x * 2 * np.pi / 64)
            ny = np.sin(y * 2 * np.pi / 64)
            nz = 1.0
            length = np.sqrt(nx*nx + ny*ny + nz*nz)
            nx, ny, nz = nx/length, ny/length, nz/length
            # Map [-1, 1] to [0, 255]
            img[y, x] = [int((nx + 1) * 127.5), int((ny + 1) * 127.5), int((nz + 1) * 127.5)]
    Image.fromarray(img).save(path)

def create_roughness(path, size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            r = int(255 * (x / size))
            img[y, x] = [r, r, r]
    Image.fromarray(img).save(path)

def create_metallic(path, size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 0] # non-metallic
    Image.fromarray(img).save(path)

if __name__ == "__main__":
    create_albedo("albedo.png")
    create_normal("normal.png")
    create_roughness("roughness.png")
    create_metallic("metallic.png")
    print("Test textures generated.")
