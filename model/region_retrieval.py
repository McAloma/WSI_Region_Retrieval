import sys, time, glob
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project")
from collections import deque, defaultdict

from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
from Image2Image_Retrieval_pipeline.src.wsi_loader import WSILoader
from Image2Image_Retrieval_pipeline.model.basic_retriever import Image2Image_Retriever_Qdrant
from Image2Image_Retrieval_pipeline.src.evaluater import Evaluater
from Image2Image_Retrieval_pipeline.test.save_utils import save_results_as_comment



class Region_Retriever():
    def __init__(self, encoder):
        self.encoder = encoder
        self.basic_retriever = Image2Image_Retriever_Qdrant(self.encoder)

    def mesh_slides(self, image):
        width, height = image.size
        width_step = 224 - (224 - width % 224) // (width // 224)
        height_step = 224 - (224 - height % 224) // (height // 224)

        image_patches = []
        for x in range(0, width-223, width_step):
            for y in range(0, height-223, height_step):
                cropped_image = image.crop((x, y, x+224, y+224))
                image_patches.append(cropped_image)

        return image_patches
    
    def single_retrieval(self, query_image, top_k=20, threshold=0.8):
        results = self.basic_retriever.retrieve(query_image, top_k, threshold)
        return [(result.score, result.payload) for result in results]   # 检索结果是相似度得分和payload的索引。
    
    def mesh_slides(self, image):
        width, height = image.size
        width_step = 224 - (224 - width % 224) // (width // 224)
        height_step = 224 - (224 - height % 224) // (height // 224)

        image_patches = []
        for x in range(0, max(1, width-223), width_step):
            for y in range(0, max(1, height-223), height_step):
                cropped_image = image.crop((x, y, x+224, y+224))
                image_patches.append(cropped_image)

        return image_patches
    
    def find_most_wsi_name(self, raw_results):
        score_hist = defaultdict(float)
        result_hist = defaultdict(list)
       
        for result in raw_results:
           for score, payload in result:
               wsi_name = payload["wsi_name"]
               score_hist[wsi_name] += score
               result_hist[wsi_name].append((score, payload))

        try:
            target = max(score_hist, key=score_hist.get)
            return target, result_hist[target]
        except:
            return None, None

    def find_region(self, target_results):
        def is_adjacent_or_overlapping(rect1, rect2):
            # 解包矩形参数
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2

            # 计算矩形1和矩形2的边界
            left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
            left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2

            # 判断是否重叠
            overlap = not (right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)

            # 判断是否相邻
            adjacent = (
                (right1 == left2 or right2 == left1) and (top1 < bottom2 and top2 < bottom1) or  # 左右相邻
                (bottom1 == top2 or bottom2 == top1) and (left1 < right2 and left2 < right1)    # 上下相邻
            )

            return overlap or adjacent
        
        rect_list = [
            (result[0], [
                int(result[1]['position'][0]) * (2 ** int(result[1]['level'])), 
                int(result[1]['position'][1]) * (2 ** int(result[1]['level'])), 
                int(result[1]['patch_size'][0]) * (2 ** int(result[1]['level'])),  
                int(result[1]['patch_size'][1]) * (2 ** int(result[1]['level'])), 
              ])
            for result in target_results
        ]   # 所有 patch 的数据均还原到 level 0 上
            
        score_results = defaultdict(list)
        region_results = defaultdict(list)

        checked_index = []
        target_deque = deque()     # 用队列来进行检索
        for i in range(len(rect_list)):
            if i in checked_index:
                continue
            target_deque.append([i, rect_list[i]])
            checked_index.append(i)

            while len(target_deque) != 0:
                index, cur = target_deque.popleft()
                score1, rect1 = cur[0], cur[1]
                
                score_results[i].append(score1)
                region_results[i].append(rect1)

                for j in range(i+1, len(rect_list)):
                    if j in checked_index:
                        continue
                
                    _, rect2 = rect_list[j]
                    if is_adjacent_or_overlapping(rect1, rect2):
                        target_deque.append([j, rect_list[j]])
                        checked_index.append(j)

        regions = []
        for key in region_results:
            cur_patches = region_results[key]
    
            result_x = min([res[0] for res in cur_patches])
            result_y = min([res[1] for res in cur_patches])
            result_w = max([res[0]+res[2] for res in cur_patches]) - result_x
            result_h = max([res[1]+res[3] for res in cur_patches]) - result_y

            target_region = [result_x, result_y, result_w, result_h]
            regions.append(target_region)

        return regions

    def redifine_region(self, target_region, ratio):
        x, y, width, height = target_region
        mid_x = x + width // 2
        mid_y = y + height // 2

        redifine_width = int((width * height * ratio) ** 0.5)   # ratio = width / heigh
        redifine_height = int(redifine_width / ratio)
        redifine_x = max(0, mid_x - redifine_width // 2)
        redifine_y = max(0, mid_y - redifine_height // 2)

        return [redifine_x, redifine_y, redifine_width, redifine_height]

    def retrieve(self, image, top_k=20, threshold=0.8):
        width, height = image.size
        image_patches = self.mesh_slides(image)
        raw_results = [self.single_retrieval(patch, top_k, threshold) for patch in image_patches]

        target_wsi_name, target_results = self.find_most_wsi_name(raw_results)   

        if not target_wsi_name:
            return None, None

        region_results = self.find_region(target_results)
        
        redifine_region = [self.redifine_region(region, width/height) for region in region_results]

        return target_wsi_name, redifine_region
    




if __name__ == "__main__":
    def get_region(loader, name, pos, size, level):
        wsi_file = f"Image2Image_Retrieval_pipeline/data/TCGA/{name}"
        wsi_path = glob.glob(f"{wsi_file}/*.svs")
        region = loader.wsi_loading_patch(wsi_path[0], pos, level, size)
        return region

    wsi_loader = WSILoader()
    wsi_name = "347d673d-e9b2-4f23-b49d-f2dc86a69efa"
    pos, size, level = (5131, 3151), (1234, 1234), 2
    query_region = get_region(wsi_loader, wsi_name, pos, size, level)
    
    # encoder = WSI_Image_UNI_Encoder()
    encoder = None                                   # testing
    retriever = Region_Retriever(encoder)
    eva = Evaluater(encoder)

    start = time.time()
    target_wsi_name, results = retriever.retrieve(query_region, threshold=0.0)     # 获得的region都是在 level 0
    end = time.time()
    print(f"Total Retrieved time: {end-start}")

    for result in results:
        x, y, w, h = result
        pos, size = (x, y), (w, h)
        retrieved_region = get_region(wsi_loader, target_wsi_name, pos, size, 0)

        if target_wsi_name == None:
            cos_sim, iou = 0, 0
        else:
            cos_sim = eva.calculate_similarity(query_region, retrieved_region)
            iou = eva.region_retrieval_self_IoU(pos, size, level, result)

        print(target_wsi_name, result, 0, cos_sim, iou)