import sys, cv2, requests
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from io import BytesIO
from PIL import Image
import numpy as np
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder


class Evaluater():
    def __init__(self, encoder=None):
        self.wsi_patch_encoder = encoder

    def load_img_url(self, img_url):
        if "http" in img_url:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(img_url).convert("RGB")

        return image

    def calculate_similarity(self, query_image, retrieval_image):
        if self.wsi_patch_encoder:
            query_embedding = self.wsi_patch_encoder.encode_image(query_image)
            retrieval_embedding = self.wsi_patch_encoder.encode_image(retrieval_image)
        else:
            query_embedding = [1.0 for _ in range(1024)]
            retrieval_embedding = [0.1 for _ in range(1024)]

        dot_product = np.dot(query_embedding, retrieval_embedding)

        norm_a = np.linalg.norm(query_embedding)
        norm_b = np.linalg.norm(retrieval_embedding)

        cosine_similarity = dot_product / (norm_a * norm_b)

        return cosine_similarity
    
    def region_retrieval_self_IoU(self, pos, size, level, result):
        q_x, q_y = pos
        q_w, q_h = size
        q_x = q_x * (2 ** level)
        q_y = q_y * (2 ** level)
        q_w = q_w * (2 ** level)
        q_h = q_h * (2 ** level)

        rect_query = ((q_x, q_y), (q_w, q_h), 0)

        r_x, r_y, r_w, r_h = result
        rect_region = ((r_x, r_y), (r_w, r_h), 0)

        inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect_query, rect_region)

        if inter_type > 0 and inter_pts is not None:
            inter_area = cv2.contourArea(inter_pts)
        else:
            inter_area = 0.0

        area1 = q_w * q_h
        area2 = r_w * r_h
        union_area = area1 + area2 - inter_area
        iou_score = inter_area / union_area

        return iou_score

    

if __name__ == "__main__":
    encoder = WSI_Image_UNI_Encoder()
    eva = Evaluater(encoder)
    query_url = "http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/241183-21.tiff/536/1024/256/512/4"
    retrieval_url = "http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/241183-21.tiff/1092/1417/886/1773/3"
    cos_sim = eva.calculate_similarity(query_url, retrieval_url)
    print(cos_sim)