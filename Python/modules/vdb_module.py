import time, re, copy, uuid
from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.models import NamedVector
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    UpdateOperation, Vector
)

METASEARCH_COLLECTION_MAPPING = {
    # 'text': {
    #     'collection_name': 'heliconsearch_text',
    #     #'vectors_config': {'size': }
    # },
    'image': {
        'collection_name': 'heliconsearch_image'
    },
    # 'video': {
    #     'collection_name': 'heliconsearch_video'
    # }
}


class HybridSearcher:
    def __init__(self, host='127.0.0.1:6333'):
        
        print("HybridSearch called host=%s" % host)
        self.llm_recall_num = 10
        self.clip_recall_num = 10
        self.host = host
        try :
            # initialize Qdrant client
            self.qdrant_client = QdrantClient(self.host)
        except Exception as e:
            print("Qdrant connection failed %s" % str(e))
        
        print("Qdrant vector database connected success")
        
        # 创建collection
        for key in METASEARCH_COLLECTION_MAPPING.keys():
            if not self.qdrant_client.collection_exists(METASEARCH_COLLECTION_MAPPING[key]['collection_name']): #creating a Collection
                print('HybridSearcher:', 'Create collection:  | %s |' % METASEARCH_COLLECTION_MAPPING[key]['collection_name'])
                if key == 'text':
                    pass
                elif key == 'image':
                    self.qdrant_client.create_collection(
                            collection_name = METASEARCH_COLLECTION_MAPPING[key]['collection_name'],
                            vectors_config={ #Named Vectors
                                #"image": models.VectorParams(size=512, distance=models.Distance.COSINE),
                                "text": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                                "ocr": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                                "asr": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                            }
                    )
                elif key == 'video':
                    pass
                else:
                    print('Error: failed to create HybridSearcher, unknown type:', key)
                    return
            else:
                print('HybridSearcher:', 'Collection exists:  | %s |' % METASEARCH_COLLECTION_MAPPING[key]['collection_name'])

    
    def get_uuid(self):
        return uuid.uuid1().hex
    
    def scored_point_to_dict(self, scored_point):
        # 将 ScoredPoint 对象转换为字典
        return {
            "id": scored_point.id,
            "version": scored_point.version,
            "score": scored_point.score,
            "payload": scored_point.payload,
            "vector": scored_point.vector,
            "shard_key": scored_point.shard_key,
            "order_value": scored_point.order_value
        }

    def merge_search_results(self, text_results, image_results):
        merged_results = {}
        # 处理 text_results
        for result in text_results:
            if result.id not in merged_results:
                merged_results[result.id] = {
                    "score": result.score,
                    "text_score": result.score,
                    "image_score": 0,  # 初始化为 0
                    "payload": result.payload,
                    "vectors": result.vector
                }
            else:
                merged_results[result.id]["text_score"] = result.score
                merged_results[result.id]["score"] += result.score

        # 处理 image_results
        for result in image_results:
            if result.id not in merged_results:
                merged_results[result.id] = {
                    "score": result.score,
                    "text_score": 0,  # 初始化为 0
                    "image_score": result.score,
                    "payload": result.payload,
                    "vectors": result.vector
                }
            else:
                merged_results[result.id]["image_score"] = result.score
                merged_results[result.id]["score"] += result.score

        # 计算新的合并分数（求和后除以 2）
        for result in merged_results.values():
            result["score"] = (result["text_score"] + result["image_score"]) / 2
        
        # 按分数降序排序
        #sorted_results = sorted(merged_results.items(), key=lambda x: x[1]["score"], reverse=True)
        sorted_results = sorted(merged_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results

    def upload(self, category, dataList):
        if category == 'image' or category == 'video':
            texts_embeded = []
            ocr_embeded = []
            for item in dataList:
                #print('item=', item['image'])
                texts_embeded.append(copy.deepcopy(item['text_vec']))
                ocr_embeded.append(copy.deepcopy(item['ocr_vec']))
                del item['text_vec']
                del item['ocr_vec']
            try:
                #print(len(texts_embeded), len(texts_embeded[0]), len(images_embeded), len(images_embeded[0]))
                self.qdrant_client.upload_points(
                    collection_name=METASEARCH_COLLECTION_MAPPING[category]['collection_name'],
                    points=[
                        models.PointStruct(
                            id=self.get_uuid(), #unique id of a point, pre-defined by the user
                            vector={
                                "text": texts_embeded[idx], #embeded caption
                                "ocr": ocr_embeded[idx], #embeded caption
                                "asr": texts_embeded[idx], #embeded caption
                            },
                            payload=doc #original image and its caption
                        )
                        for idx, doc in enumerate(dataList)
                    ]
                )
               
                print('#####数据库已存档:', item['meeting_name'], '(session_id=%s)' % item['session_id'])
            except Exception as e:
                print("upload points failed", str(e), "points=", item)
                #TODO:

    
    # 单向量搜索 - 修改为：text或者image
    # mode - 0-混合模式， 1-文本模式， 2-ocr模式, 3-asr模式
    def search_by_vector(self, session_id, queryList, category='image', limit=10, mode=0):
        ans = []
        query_results = []
        #vectored_name = 'text'
        if mode == 1:
            vectored_name = 'text'
        elif mode == 2:
            vectored_name = 'ocr'
        elif mode == 3:
            vectored_name = 'asr'
        if mode not in [0,1,2,3]:
            return ans
        # 定义过滤条件
        filter = Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id)
                )
            ]
        )
        if category == 'image' and mode == 0:
            search_result = None
            for item in queryList:
                vector = item['qv']
                search_result = self.qdrant_client.query_points(
                    collection_name=METASEARCH_COLLECTION_MAPPING[category]['collection_name'],
                    prefetch=[
                        models.Prefetch(
                            query=vector,
                            using="text",
                            limit=limit,
                        ),
                        models.Prefetch(
                            query=vector,
                            using="ocr",
                            limit=limit,
                        ),
                        models.Prefetch(
                            query=vector,
                            using="asr",
                            limit=limit,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    with_payload=True,
                    query_filter=filter
                )
                # print('search_result type=', type(search_result))
                # print('result=', search_result)
                query_results.append(search_result)
                # vector result
                query_results = query_results
                for num, search_result in enumerate(query_results):
                    #print('\n查询:', queryList[num]['query'],'搜索结果: %s 张' % len(search_result), 'limit=', limit)
                    result = []
                    for idx, item in enumerate(search_result):
                        # print('item type=', type(item))
                        # print('itme=',item)
                        #print('------------------------------------------------------------------------------------------------')
                        for point in item[1]:    
                            record = {}
                            point = self.scored_point_to_dict(point)
                            #print('第%s张:' % (idx+1), '分数:', item['score'])
                            #print('名称:', item['payload']['file'])
                            record['meeting_name'] = point['payload']['meeting_name']
                            record['session_id'] = point['payload']['session_id']
                            record['file'] = point['payload']['file']
                            record['frame_id'] = point['payload']['frame_id']
                            record['start'] = point['payload']['start']
                            record['end'] = point['payload']['end']
                            record['desc'] = point['payload']['desc']
                            record['asr'] = point['payload']['asr']
                            record['ocr'] = point['payload']['ocr']
                            record['ocr_tks'] = point['payload']['ocr_tks']
                            record['score'] = point['score']
                            result.append(record)
                    ans.append(result)
        elif category == 'image' and mode in [1, 2, 3]:
            #find_image = self.embeddingFactory.get_text_embeddings(self.embeddingFactory.text_model, self.embeddingFactory.text_tokenizer, queryList) 
            search_result = None
            for item in queryList:
                vector = item['qv']
                #print('query', len(vector), type(vector), vector)
                search_result = self.qdrant_client.search(
                    collection_name=METASEARCH_COLLECTION_MAPPING[category]['collection_name'], #searching in our collection
                    query_vector=NamedVector(name=vectored_name, vector=vector), #searching only among image vectors with our textual query
                    with_payload=True, #user-readable information about search results, we are interested to see which image we will find
                    limit=limit, #top-5 similar to the query result
                    query_filter=filter
                )
                #print('search_result type=', type(search_result))
                query_results.append(search_result)

            # vector result
            query_results = query_results
            for num, search_result in enumerate(query_results):
                #print('\n查询:', queryList[num]['query'],'搜索结果: %s 张' % len(search_result), 'limit=', limit)
                result = []
                for idx, item in enumerate(search_result):
                    # print('item type=', type(item))
                    # print('itme=',item)
                    #print('------------------------------------------------------------------------------------------------') 
                    record = {}
                    point = self.scored_point_to_dict(item)
                    #print('第%s张:' % (idx+1), '分数:', item['score'])
                    #print('名称:', item['payload']['file'])
                    record['meeting_name'] = point['payload']['meeting_name']
                    record['session_id'] = point['payload']['session_id']
                    record['file'] = point['payload']['file']
                    record['frame_id'] = point['payload']['frame_id']
                    record['start'] = point['payload']['start']
                    record['end'] = point['payload']['end']
                    record['desc'] = point['payload']['desc']
                    record['asr'] = point['payload']['asr']
                    record['ocr'] = point['payload']['ocr']
                    record['ocr_tks'] = point['payload']['ocr_tks']
                    record['score'] = point['score']
                    result.append(record)
                ans.append(result)
            #print('------------------------------------------------------------------------------------------------')
        if ans:
            return ans[0]
        else:
            return ans

    # 根据doc_id搜索
    def search_by_sessionid(self, session_id, category='image'):
        search_result = self.qdrant_client.scroll(
            collection_name=METASEARCH_COLLECTION_MAPPING[category]['collection_name'],
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            limit=100000
        )
        final_list = []
        for idx, item in enumerate(search_result[0]):
            payload = item.payload
            final_list.append(payload)

        return final_list
    

    def update_points(self, _session_id, _frame, asr_vec, category='image'):

        # 根据字符串和 int 值筛选点
        filter = models.Filter(
            must=[
                models.FieldCondition(key="session_id", match=models.MatchValue(value=_session_id)),
                models.FieldCondition(key="frame_id", match=models.MatchValue(value=_frame['frame_id']))
            ]
        )

        # 搜索符合条件的点
        search_result = self.qdrant_client.scroll(
            collection_name=METASEARCH_COLLECTION_MAPPING[category]['collection_name'],
            scroll_filter=filter
        )

        # 获取符合条件的点 ID
        points_to_update = [point.id for point in search_result[0]]
        if not points_to_update:
            return False

        self.qdrant_client.batch_update_points(
            collection_name=METASEARCH_COLLECTION_MAPPING[category]['collection_name'],
            update_operations=[
                # models.UpsertOperation(
                #     upsert=models.PointsList(
                #         points=[
                #             models.PointStruct(
                #                 id=1,
                #                 vector=[1.0, 2.0, 3.0, 4.0],
                #                 payload={},
                #             ),
                #         ]
                #     )
                # ),
                models.UpdateVectorsOperation(
                    update_vectors=models.UpdateVectors(
                        points=[
                            models.PointVectors(
                                id=points_to_update[0],
                                vector={"asr": asr_vec},
                            )
                        ]
                    )
                ),
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={
                            "asr": _frame['asr']
                        },
                        points=points_to_update,
                    )
                ),
            ],
        )

        print(f"[ASR更新成功]: session = {_session_id} 帧号 = {_frame['frame_id']} length = {len(_frame['asr'])}")

        return True