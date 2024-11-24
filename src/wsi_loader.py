import sys, requests, os
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import numpy as np
from io import BytesIO
from PIL import Image
from multiprocessing import shared_memory
from openslide import OpenSlide


class WSILoader():
    def wsi_loading_patch(self, wsi_path, position=(0,0), level=0, size=(224,224), show_info=False):
        slide = OpenSlide(wsi_path)

        if show_info:
            print(f"文件名: {wsi_path}")
            print(f"图像宽度: {slide.dimensions[0]} 像素")
            print(f"图像高度: {slide.dimensions[1]} 像素")
            print(f"级别数: {slide.level_count}")
            print(f"每级别的尺寸: {slide.level_dimensions}")
            print(f"每级别的降采样因子: {slide.level_downsamples}")

        ture_pos = tuple([int(pos*slide.level_downsamples[level]) for pos in position])
        image = slide.read_region(ture_pos, level, size)
        slide.close()
        image = image.convert('RGB')

        return image

    def wsi_loading_patches(self, wsi_path):
        """Get WSI infos and patches from WSI path"""
        slide = OpenSlide(wsi_path)
        levels, patch_size = slide.level_count, (224, 224)

        loaded_infos, loaded_images = [], []
        for level in range(2, slide.level_count):
            ratio = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level][0], slide.level_dimensions[level][1]
            for w in range(0, width, patch_size[0]):
                for h in range(0, height, patch_size[0]):
                    ture_pos = (int(w * ratio), int(h * ratio))

                    infos = {
                        "wsi_name":wsi_path.split("/")[-1],
                        "position":ture_pos,    # basic on level 0
                        "level":level,
                        "size":patch_size,
                    }

                    image = slide.read_region(ture_pos, level, patch_size)
                    image = image.convert('RGB')
                    loaded_images.append(image)

        return loaded_infos, loaded_images 

    def get_wsi_shared_patches(self, wsi_path):
        """ loading patched image in share memory"""
        loaded_infos, loaded_images = self.wsi_loading_patches(wsi_path)
        img_arrays = np.array([np.array(img, dtype=np.uint8) for img in loaded_images])

        shm = shared_memory.SharedMemory(create=True, size=img_arrays.nbytes)
        shared_array = np.ndarray(img_arrays.shape, dtype=img_arrays.dtype, buffer=shm.buf)
        shared_array[:] = img_arrays[:]     # 将图像数据复制到共享内存

        return loaded_infos, shm, img_arrays.shape, img_arrays.dtype


def load_wsi_region(wsi_url, x, y, w, h, level, angle):
    """加载 region 的时候用的是中心点和角度值。"""
    x, y, w, h, level, angle = int(x), int(y), int(w), int(h), int(level), int(angle)
    length = max(w, h)
    background_url = f"{wsi_url}/{x-length}/{y-length}/{2*length}/{2*length}/{level}"

    response = requests.get(background_url)
    background = response.content
    background = Image.open(BytesIO(background)).convert("RGB")

    rotated_image = background.rotate(-angle, center=(length, length), expand=False)
    left = length - w / 2
    upper = length - h / 2
    right = length + w / 2
    lower = length + h / 2
    cropped_image = rotated_image.crop((left, upper, right, lower))
    
    return background, cropped_image



if __name__ == "__main__":
    wsi_name = "347d673d-e9b2-4f23-b49d-f2dc86a69efa"
    wsi_path = f"Image2Image_Retrieval_pipeline/data/TCGA/{wsi_name}"
    wsi_loader = WSILoader()

    position, level, size = (0,0), 9, (200,200)
    wsi_patch_image = wsi_loader.wsi_loading_patch(wsi_path, position, level, size)
    wsi_patch_image.save("MDI_RAG_Image2Image_Research/data/cache/"+wsi_path.split("/")[-1].split(".")[0]+".png")

    position, level, size = (200,200), 8, (200, 200)
    wsi_patch_image = wsi_loader.wsi_loading_patch(wsi_path, position, level, size)
    wsi_patch_image.save("MDI_RAG_Image2Image_Research/data/cache/"+wsi_path.split("/")[-1].split(".")[0]+"1.png")