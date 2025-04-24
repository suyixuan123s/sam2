import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# from pycococreatortools import pycococreatortools
#
# image_info = pycococreatortools.create_image_info(image_id, image_filename, image.size)

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def apply_color_mask(image, mask, color, color_dark = 0.5):#对掩体进行赋予颜色
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - color_dark) + color_dark * color[c], image[:, :, c])
    return image

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1)

    ax.imshow(img)

image = Image.open('/home/suyixuan/AI/Pose_Estimation/sam2/notebooks/images/truck.jpg')
image = np.array(image.convert("RGB"))
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2//configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show()
image_select = image.copy()
for i in range(len(masks)):
    color = tuple(np.random.randint(0, 256, 3).tolist())#随机列表颜色，就是
    selected_mask=masks[i]['segmentation']
    selected_image = apply_color_mask(image_select,selected_mask, color)
cv2.imwrite("data/res.jpg", selected_image)