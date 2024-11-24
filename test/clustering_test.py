import os, sys, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project")
import numpy as np
from tqdm import tqdm
from openslide import OpenSlide
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
from Image2Image_Retrieval_pipeline.src.wsi_loader import WSILoader
from Image2Image_Retrieval_pipeline.model.basic_retriever import Image2Image_Retriever_Qdrant
from Image2Image_Retrieval_pipeline.model.region_retrieval import Region_Retriever
from Image2Image_Retrieval_pipeline.src.evaluater import Evaluater
from Image2Image_Retrieval_pipeline.test.save_utils import save_results_as_comment

random.seed(2024)



class Region_Retriever_diff_cluster(Region_Retriever):
    def __init__(self, encoder):
        self.encoder = encoder
        self.basic_retriever = Image2Image_Retriever_Qdrant(self.encoder)

    def kmeams_anchor_selection(self, patches, n_clusters=4, using_pp=False):
        # 提取特征
        features = [self.encoder(image) for image in patches]
        features = np.array(features)  # 转换为 NumPy 数组

        # KMeans 聚类
        if using_pp:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_  # 每个图像的标签
        centers = kmeans.cluster_centers_  # 每个类的中心点

        # 找出每个类中心最近的图像
        center_images = []
        for cluster_idx in range(n_clusters):
            # 找出属于当前类的图像索引
            cluster_indices = np.where(labels == cluster_idx)[0]
            
            # 计算每个图像到类中心的距离
            cluster_features = features[cluster_indices]
            distances = np.linalg.norm(cluster_features - centers[cluster_idx], axis=1)
            
            # 找到距离最小的图像
            closest_image_idx = cluster_indices[np.argmin(distances)]
            center_images.append(patches[closest_image_idx])

        return center_images

    def gmm_cluster_and_select_centers(self, image_list, n_clusters=4):
        # 提取特征
        features = [self.encoder(image) for image in image_list]
        features = np.array(features)  # 转换为 NumPy 数组

        # 使用 GMM 进行聚类（EM 算法）
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(features)
        labels = gmm.predict(features)  # 每个图像的标签
        centers = gmm.means_  # 每个类的中心点（均值）

        # 找出每个类中心最近的图像
        center_images = []
        for cluster_idx in range(n_clusters):
            # 找出属于当前类的图像索引
            cluster_indices = np.where(labels == cluster_idx)[0]
            
            # 计算每个图像到类中心的距离
            cluster_features = features[cluster_indices]
            distances = np.linalg.norm(cluster_features - centers[cluster_idx], axis=1)
            
            # 找到距离最小的图像
            closest_image_idx = cluster_indices[np.argmin(distances)]
            center_images.append(image_list[closest_image_idx])

        return center_images

    def retrieve(self, image, clustering_method=None, top_k=20, threshold=0.8):
        width, height = image.size
        image_patches = self.mesh_slides(image)

        if clustering_method == "kmeans":
            select_patches = self.kmeams_anchor_selection(image_patches)
        elif clustering_method == "kmeanspp":
            select_patches = self.kmeams_anchor_selection(image_patches, using_pp=True)
        elif clustering_method == "em":
            select_patches = self.gmm_cluster_and_select_centers(image_patches)
        else:
            select_patches = image_patches

        raw_results = [self.single_retrieval(patch, top_k, threshold) for patch in select_patches]

        target_wsi_name, target_results = self.find_most_wsi_name(raw_results)   

        if not target_wsi_name:
            return None, None

        region_results = self.find_region(target_results)
        
        redifine_region = [self.redifine_region(region, width/height) for region in region_results]

        return target_wsi_name, redifine_region

if __name__ == "__main__":
    from Image2Image_Retrieval_pipeline.test.threshold_test import main, get_query_infos

    folder_path = "Image2Image_Retrieval_pipeline/data/patches"  
    all_files_and_dirs = os.listdir(folder_path)
    target_names = [d for d in all_files_and_dirs if os.path.isdir(os.path.join(folder_path, d))]
    selected_names = random.sample(target_names, 5)

    query_region_infos = get_query_infos(selected_names)

    encoder = WSI_Image_UNI_Encoder()
    retriever = Region_Retriever_diff_cluster(encoder)
    eva = Evaluater(encoder)
    wsi_loader = WSILoader()

    clustering_methods = [None, "kmeans", "kmeanspp", "em"]
    for method in clustering_methods:
        cos_sim, iou, times = main(query_region_infos, retriever, eva, wsi_loader)

        print(clustering_methods, cos_sim, iou, times)
        cluster_centers = [clustering_methods, cos_sim, iou, times]
        save_results_as_comment("example.py", cluster_centers)