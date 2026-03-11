import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt

def read_exr(path):
    file = OpenEXR.InputFile(path)
    dw = file.header()['displayWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    # Read Depth channel
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = file.channel('Depth', FLOAT)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size)
    return depth

depth_map = read_exr('test_out/gbuffer.exr')

print(f"Min Depth: {np.min(depth_map[depth_map < 1000])}m")
print(f"Max Depth: {np.max(depth_map[depth_map < 1000])}m")

plt.imshow(depth_map, cmap='magma', vmax=100) # Cap at 100m for visibility
plt.colorbar(label='Metres')
plt.title('Ground Truth Depth Map')
plt.show()