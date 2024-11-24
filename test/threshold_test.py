import os, sys, time, glob, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project")
import numpy as np
from tqdm import tqdm
from openslide import OpenSlide

from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
from Image2Image_Retrieval_pipeline.src.wsi_loader import WSILoader
from Image2Image_Retrieval_pipeline.model.region_retrieval import Region_Retriever
from Image2Image_Retrieval_pipeline.src.evaluater import Evaluater
from Image2Image_Retrieval_pipeline.test.save_utils import save_results_as_comment

random.seed(2024)


def get_region(loader, name, pos, size, level):
    wsi_file = f"Image2Image_Retrieval_pipeline/data/TCGA/{name}"
    wsi_path = glob.glob(f"{wsi_file}/*.svs")
    region = loader.wsi_loading_patch(wsi_path[0], pos, level, size)
    return region

def get_query_infos(target_names, n=5, bound=50, show_info=False):
    query_region_infos = []
    for name in target_names:
        wsi_file = f"Image2Image_Retrieval_pipeline/data/TCGA/{name}"
        wsi_path = glob.glob(f"{wsi_file}/*.svs")[0]
        slide = OpenSlide(wsi_path)

        if show_info:
            print(f"文件名: {wsi_path}")
            print(f"图像宽度: {slide.dimensions[0]} 像素")
            print(f"图像高度: {slide.dimensions[1]} 像素")
            print(f"级别数: {slide.level_count}")
            print(f"每级别的尺寸: {slide.level_dimensions}")
            print(f"每级别的降采样因子: {slide.level_downsamples}")

        level = slide.level_count
        sizes = slide.level_dimensions

        for l in range(level):
            for _ in range(n):
                size = sizes[l]

                w = h = 0
                while w < 224 or h < 224:
                    x = random.randint(bound, size[0]-bound)
                    y = random.randint(bound, size[1]-bound)
                    w = min(1792, random.randint(x, size[0]-bound))
                    h = min(1792, random.randint(y, size[1]-bound))     # 224 * 8

                query_info = (name, (x, y), (w, h), l)
                query_region_infos.append(query_info)

    return query_region_infos

def main(query_region_infos, retriever, eva, wsi_loader, threshold=0.8, top_k=20, clustering_method=None):
    cos_sim_list, iou_list, time_list = [], [], []

    for info in tqdm(query_region_infos, ascii=True):
        wsi_name, pos, size, level = info
        query_region = get_region(wsi_loader, wsi_name, pos, size, level)

        start = time.time()
        if clustering_method:
            target_wsi_name, results = retriever.retrieve(query_region, clustering_method, top_k, threshold=threshold) 
        else:
            target_wsi_name, results = retriever.retrieve(query_region, top_k, threshold=threshold) 
        end = time.time()

        if not target_wsi_name:
            cos_sim_list.append(0)
            iou_list.append(0)
            continue

        for result in results:
            x, y, w, h = result
            pos, size = (x, y), (w, h)
            retrieved_region = get_region(wsi_loader, target_wsi_name, pos, size, 0)
            cos_sim = eva.calculate_similarity(query_region, retrieved_region)
            iou = eva.region_retrieval_self_IoU(pos, size, level, result)

            cos_sim_list.append(cos_sim)
            iou_list.append(iou)

        time_list.append(start-end)

    print(cos_sim_list, iou_list, time_list)

    return np.mean(cos_sim_list), np.mean(iou_list), np.mean(time_list)



if __name__ == "__main__":
    folder_path = "Image2Image_Retrieval_pipeline/data/patches"  
    all_files_and_dirs = os.listdir(folder_path)
    target_names = [d for d in all_files_and_dirs if os.path.isdir(os.path.join(folder_path, d))]
    selected_names = random.sample(target_names, 5)

    query_region_infos = get_query_infos(selected_names)

    encoder = WSI_Image_UNI_Encoder()
    # encoder = None
    retriever = Region_Retriever(encoder)
    eva = Evaluater(encoder)
    wsi_loader = WSILoader()

    thresholds = [0.7, 0.8, 0.85, 0.9]
    for threshold in thresholds:
        cos_sim, iou, times = main(query_region_infos, retriever, eva, wsi_loader, threshold=threshold)

        print(20, threshold, cos_sim, iou, times)
        cluster_centers = [20, threshold, cos_sim, iou, times]
        save_results_as_comment("Image2Image_Retrieval_pipeline/test/threshold_test.py", cluster_centers)

    top_k = [10, 20, 30]
    for threshold in thresholds:
        cos_sim, iou, times = main(query_region_infos, retriever, eva, wsi_loader, top_k=top_k)

        print(top_k, 0.8, cos_sim, iou, times)
        cluster_centers = [top_k, 0.8, cos_sim, iou, times]
        save_results_as_comment("Image2Image_Retrieval_pipeline/test/threshold_test.py", cluster_centers)