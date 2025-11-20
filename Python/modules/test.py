from langchain_community.embeddings import OpenVINOBgeEmbeddings

embedding_model_name = './models/bge-m3'#npu_embedding_dir if USING_NPU else embedding_model_id.value
batch_size = 4#1 if USING_NPU else 4
embedding_model_kwargs = {"device": "CPU", "compile": False}
encode_kwargs = {
    "mean_pooling": embedding_model_configuration["mean_pooling"],
    "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
    "batch_size": batch_size,
}

embedding = OpenVINOBgeEmbeddings(
    model_name_or_path=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs,
)
if USING_NPU:
    embedding.ov_model.reshape(1, 512)
embedding.ov_model.compile()

text = "This is a test document."
embedding_result = embedding.embed_query(text)
embedding_result[:3]