import os, sys, timm, json, torch, asyncio
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

from Image2Image_Retrieval_pipeline.src.wsi_background import load_wsi_thumbnail, get_patch_background_ratio

    

class CustomWSIDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class WSIUNIEncoder():
    def __init__(self, **kwargs):
        self.embed_model =  timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        local_dir = "checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
        self._device = self.infer_torch_device()
        print(self._device)
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), 
                                         map_location="cpu", 
                                         weights_only=True), strict=True)
        self.embed_model = self.embed_model.to(self._device)
        self.embed_model.eval()

    def infer_torch_device(self):
        """Infer the input to torch.device."""
        try:
            has_cuda = torch.cuda.is_available()
        except NameError:
            import torch  # pants: no-infer-dep
            has_cuda = torch.cuda.is_available()
        if has_cuda:
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode_wsi_patch(self, wsi_name, dataloader):
        embeddings = []
        with torch.no_grad():
            for images in tqdm(dataloader, desc=f"WSI name: {wsi_name}", ascii=True):
                images = images.to(self._device)
                embedding = self.embed_model(images)
                embeddings.append(embedding.cpu())

        if embeddings == []:
            return []
        else:
            patch_embeddings = torch.cat(embeddings, dim=0).cpu().tolist()
            return patch_embeddings


class Embedding_loader():
    def __init__(self, ratio=1):
        self.wsi_patch_encoder = WSIUNIEncoder()

        self.cache_path = "Image2Image_Retrieval_pipeline/data/embeddings"
        if ratio < 1:
             self.cache_path = f"Image2Image_Retrieval_pipeline/data/embeddings_{ratio}"
             
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.loaded_embeddings = os.listdir(self.cache_path)
        self.ratio = ratio

    async def load_image_paths(self, folder_path):
        """异步加载图像路径"""
        image_paths = []
        filenames = os.listdir(folder_path)
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
        return image_paths

    async def loading_wsi_image(self, wsi_name):
        """在 CPU 上获取 WSI patch 的 Dataloader。"""
        folder_path = os.path.join("Image2Image_Retrieval_pipeline/data/patches", wsi_name)
        patch_infos = os.listdir(folder_path)
        image_paths = await self.load_image_paths(folder_path)
        
        thumbnail, num_level = load_wsi_thumbnail(wsi_name)
        infos, loaded_image_paths = [], []

        for info, path in zip(patch_infos, image_paths):

            info = info[:-4].split("_")
            level = info.pop()
            wsi_info = (info[0], info[1], 224, 224, level[1:])
            
            _, white_pixel_ratio = get_patch_background_ratio(thumbnail, num_level, wsi_info)

            if white_pixel_ratio < self.ratio:
                infos.append(info)
                loaded_image_paths.append(path)

        wsi_dataset = CustomWSIDataset(loaded_image_paths, self.wsi_patch_encoder.transform)
        dataloader = DataLoader(wsi_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

        return infos, dataloader

    def loading_worker(self, input_queue, output_queue):
        while True:
            wsi_name = input_queue.get()
            if wsi_name is None:
                break

            if wsi_name in self.loaded_embeddings:
                print(f"WSI {wsi_name} cached.")
                output_queue.put((wsi_name, [], []))
            else:
                patch_infos, dataloader = asyncio.run(self.loading_wsi_image(wsi_name))
                output_queue.put((wsi_name, patch_infos, dataloader))

    def encoding_worker(self, input_queue):
        while True:
            item = input_queue.get()
            if item is None:
                break

            wsi_name, patch_infos, dataloader = item
            patch_embeddings = self.wsi_patch_encoder.encode_wsi_patch(wsi_name, dataloader)

            dir_path = os.path.join(self.cache_path, wsi_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            info_path = os.path.join(self.cache_path, wsi_name, "patch_info.json")
            with open(info_path, 'w') as file:
                json.dump(patch_infos, file)

            embedding_path = os.path.join(self.cache_path, wsi_name, "embeddings.json")
            with open(embedding_path, 'w') as file:
                json.dump(patch_embeddings, file)

    def main(self, wsi_names_list):
        load_workers = 2
        load_queue = mp.Queue(maxsize=8)
        encode_queue = mp.Queue(maxsize=8)

        loading_processes = [mp.Process(target=self.loading_worker, args=(load_queue, encode_queue)) for _ in range(load_workers)]
        encoding_process = mp.Process(target=self.encoding_worker, args=(encode_queue,))

        for p in loading_processes:
            p.start()
        encoding_process.start()

        for wsi_name in wsi_names_list:
            load_queue.put(wsi_name)

        for _ in range(load_workers):
            load_queue.put(None)
        for p in loading_processes:
            p.join()

        encode_queue.put(None)
        encoding_process.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')

    ratios = [0.98, 0.95, 0.92, 0.89, 0.86]
    for ratio in ratios:
        loader = Embedding_loader(ratio=ratio)
        loaded_patches = "Image2Image_Retrieval_pipeline/data/patches"
        wsi_names_list = [f for f in os.listdir(loaded_patches) if os.path.isdir(os.path.join(loaded_patches, f))]
        loader.main(wsi_names_list)

    # loader = Embedding_loader(ratio=0.86)
    # loaded_patches = "Image2Image_Retrieval_pipeline/data/patches"
    # wsi_names_list = [f for f in os.listdir(loaded_patches) if os.path.isdir(os.path.join(loaded_patches, f))]
    # loader.main(wsi_names_list)