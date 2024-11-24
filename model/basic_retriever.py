import sys, qdrant_client
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/")
from PIL import Image
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder


class Image2Image_Retriever_Qdrant():
    def __init__(self, encoder):
        self.image_client_name = "WSI_Region_Retrieval"
        database_path = "Image2Image_Retrieval_pipeline/data/vector_database"  
        self.image_client = qdrant_client.QdrantClient(path=database_path)
        nums = self.image_client.count(collection_name=self.image_client_name)
        print("Number of vectors:", nums)

        # self.image_encoder = encoder

    def retrieve(self, image, top_k=20, threshold=0.8):
        # query_embedding = self.image_encoder.encode_image(image)    # 1024
        query_embedding = [1.0 for _ in range(1024)]

        retrieval_results = self.image_client.search(
            collection_name=self.image_client_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=threshold,
        )

        return retrieval_results


if __name__ == "__main__":
    encoder = WSI_Image_UNI_Encoder()
    retriever = Image2Image_Retriever_Qdrant(encoder)

    query_img_path = "Image2Image_Retrieval_pipeline/data/patches/9636dac6-5a48-4da4-aae5-81614e50d918/5376_3584_L1.png"
    query_image = Image.open(query_img_path).convert("RGB")

    results = retriever.retrieve(query_image, top_k=20)
    for result in results:
        res = result.payload
        print(res)