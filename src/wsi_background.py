import sys, requests, cv2, os, glob
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project")
from io import BytesIO
from PIL import Image
import numpy as np
from openslide import OpenSlide

from Image2Image_Retrieval_pipeline.src.wsi_loader import WSILoader


def load_wsi_thumbnail(wsi_name):
    wsi_doc_path = os.path.join("Image2Image_Retrieval_pipeline/data/TCGA", wsi_name)
    wsi_path = glob.glob(f"{wsi_doc_path}/*.svs")[0]
    slide = OpenSlide(wsi_path)

    level = slide.level_count
    width, height = slide.level_dimensions[level - 1]

    img = slide.read_region((0,0), level-1, (width, height))
    slide.close()
    img = img.convert('RGB')

    return img, level

def WSI_background_detect(wsi_image):
    """ 用 OTSU 来确定二值化阈值 """
    gray_image = wsi_image.convert("L")
    image_np = np.array(gray_image)
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    _, binary_image = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image_pil = Image.fromarray(binary_image)

    return binary_image_pil

def get_patch_background_ratio(thumbnail, num_level, info_list):
    background = WSI_background_detect(thumbnail)

    ratio = 2 ** (int(num_level) - int(info_list[4]) - 1)

    x, y = int(info_list[0]) // ratio, int(info_list[1]) // ratio
    w, h = int(info_list[2]) // ratio, int(info_list[3]) // ratio

    patch_background = background.crop((x, y, x + w, y + h))
    pixels = list(patch_background.getdata())

    white_pixel_ratio = pixels.count(255) / len(pixels)
    return patch_background, white_pixel_ratio


if __name__ == "__main__":
    wsi_name = "df1ec5b2-f0ce-48e7-a5e0-a32a27bb6e15"
    folder_path = os.path.join("Image2Image_Retrieval_pipeline/data/patches", wsi_name)
    patch_infos = os.listdir(folder_path)

    thumbnail, num_level = load_wsi_thumbnail(wsi_name)
    background = WSI_background_detect(thumbnail)
    print(type(thumbnail), type(background))
    thumbnail.save("Image2Image_Retrieval_pipeline/image/thumbnail.png")
    background.save("Image2Image_Retrieval_pipeline/image/background.png")

    loader = WSILoader()
    wsi_doc_path = os.path.join("Image2Image_Retrieval_pipeline/data/TCGA", wsi_name)
    wsi_path = glob.glob(f"{wsi_doc_path}/*.svs")[0]
    img = loader.wsi_loading_patch(wsi_path, (345, 456), num_level-1, (224, 224))
    img.save("Image2Image_Retrieval_pipeline/image/patch.png")

    wsi_info = (345, 456, 224, 224, num_level-1)
    patch_background, white_pixel_ratio = get_patch_background_ratio(thumbnail, num_level, wsi_info)
    patch_background.save("Image2Image_Retrieval_pipeline/image/patch_background.png")
    print(f"Background Ratio: {white_pixel_ratio}")