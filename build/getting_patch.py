import os, sys, json, asyncio, aiohttp, requests, logging, glob
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/")
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from openslide import OpenSlide




class ImagePatchDownloader:
    def __init__(self, max_concurrent_downloads=100):
        file_path = "Image2Image_Retrieval_pipeline/data/patches"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        self.slice_size = (224, 224)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.loaded_wsi_name_path = "Image2Image_Retrieval_pipeline/data/patches/loaded_wsis.json"
        self.image_names = self.load_wsi_name(self.loaded_wsi_name_path)

    def load_wsi_name(self, json_file_path):
        if not os.path.exists(json_file_path):
            with open(json_file_path, "w") as file:
                json.dump([], file)
            return []

        with open(json_file_path, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []

    def check_image_name(self, wsi_name):
        json_file_path = "Image2Image_Retrieval_pipeline/data/patches/loaded_wsis.json"
        image_names = self.load_wsi_name(json_file_path)
        if wsi_name not in image_names:
            image_names.append(wsi_name)
            with open(json_file_path, "w") as file:
                json.dump(image_names, file, indent=4)
            return False
        else:
            return True

    def loading_wsi(self, wsi_names):
        for wsi_name in wsi_names:
            if wsi_name in self.image_names:
                print(f"Patch of WSI {wsi_name} in the Cache.")
                return

            wsi_doc_path = os.path.join("Image2Image_Retrieval_pipeline/data/TCGA", wsi_name)
            wsi_path = glob.glob(f"{wsi_doc_path}/*.svs")[0]
            slide = OpenSlide(wsi_path)

            print(f"文件名: {wsi_path}")
            print(f"图像宽度: {slide.dimensions[0]} 像素")
            print(f"图像高度: {slide.dimensions[1]} 像素")
            print(f"级别数: {slide.level_count}")
            print(f"每级别的尺寸: {slide.level_dimensions}")
            print(f"每级别的降采样因子: {slide.level_downsamples}")

            asyncio.run(self.get_patches_async(slide, wsi_name))

            self.image_names.append(wsi_name)
            with open(self.loaded_wsi_name_path, "w") as file:
                json.dump(self.image_names, file, indent=4)

    async def get_patches_async(self, slide, wsi_name):
        levels = slide.level_count
        patch_size = self.slice_size
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

        tasks = []
        for level in range(1, levels): 
            ratio = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level]

            for w in range(0, width, patch_size[0]):
                for h in range(0, height, patch_size[1]):
                    true_pos = (int(w * ratio), int(h * ratio))
                    tasks.append(self.load_patch(slide, true_pos, level, patch_size, wsi_name, semaphore))

        with tqdm(total=len(tasks), desc=f"Processing patches for {wsi_name}") as pbar:
            for future in asyncio.as_completed(tasks):
                await future  # 确保每个协程正确执行
                pbar.update(1)
                pbar.refresh() 

    async def load_patch(self, slide, true_pos, level, patch_size, wsi_name, semaphore):
        async with semaphore:  # 控制并发任务数
            patch = slide.read_region(true_pos, level, patch_size)
            patch = patch.convert('RGB')

            # 保存图像
            patch_save_dir = f"Image2Image_Retrieval_pipeline/data/patches/{wsi_name}"
            os.makedirs(patch_save_dir, exist_ok=True)
            patch_filename = os.path.join(patch_save_dir, f"{true_pos[0]}_{true_pos[1]}_L{level}.png")
            patch.save(patch_filename)


if __name__ == "__main__":
    downloader = ImagePatchDownloader(max_concurrent_downloads=100)
    wsi_names = os.listdir("Image2Image_Retrieval_pipeline/data/TCGA")
    # wsi_names = [
    #     "df1ec5b2-f0ce-48e7-a5e0-a32a27bb6e15"
    # ]
    downloader.loading_wsi(wsi_names)