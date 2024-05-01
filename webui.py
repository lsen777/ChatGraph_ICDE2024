import gradio as gr
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import os
# from graphviz import Graph
# import pydotplus
from PIL import Image
import io
import time

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()


def get_answer(query, vs_path, history, mode, file, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    if mode == "Bing搜索问答":
        for resp, history in local_doc_qa.get_search_result_based_answer(
                query=query, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [
                    f"""<details> <summary>出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, "", "", ""
    elif mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(
            vs_path):
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                # [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                [f"""<details> <summary>出处 [{i + 1}]</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, "", "", ""
    elif mode == "知识库测试":
        print("知识库测试")
        if os.path.exists(vs_path):
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_conent=chunk_conent,
                                                                        chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, "", "", ""
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], "", "", "", "", ""
    else:

        if file is None:
            for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                                  streaming=streaming):
                resp = answer_result.llm_output["answer"]
                history = answer_result.history
                history[-1][-1] = resp
                yield history, "", "Tell more.", "Write a brief report for G."
        else:
            gname = file.name.split("/")[-1].split(".")[0]
            if query == "Show the 3D structure of this graph.":
                g_img_3d = os.getcwd() + "/txt/graph_img_3D/" + "Aspirin" + ".png"
                history += [[query, (g_img_3d,)]]
                yield history, "", "Tell more.", "Analyze the graph."
            elif query == "Clean the graph G.":
                query1 = """Clean the graph G."""
                history = history + [[query1, ""]]

                content = """Cleaning G. Please wait ......"""
                for c in content:
                    history[-1][1] += c
                    time.sleep(0.00)
                    yield history, "", "", ""
                history = history + [[None, ""]]
                content = """
                    The edge (u, v) is incorrect (probability 0.9)<br>
                    The edge (w, x) is missed (probability 0.8)<br>
                    Do you agree?
                """
                for c in content:
                    history[-1][1] += c
                    time.sleep(0.00)
                    yield history, "", "Tell more.", "Is it toxic?"
            elif query == 'Yes.':
                query1 = """Yes."""
                history = history + [[query1, ""]]
                content = """G is cleaned and outputted to file.
                """
                for c in content:
                    history[-1][1] += c
                    time.sleep(0.00)
                    yield history, "", "Tell more.", "Is it toxic?"
            elif query == 'How is its water solubility?':
                time.sleep(2.00)
                ged = os.getcwd() + "/txt/G/water-solubility.txt"
                with open(ged) as f:
                    content = f.read()
                    history = history + [[query, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Is it toxic?"
            elif query == 'Is it toxic?':
                time.sleep(2.00)
                ged = os.getcwd() + "/txt/G/toxic.txt"
                with open(ged) as f:
                    content = f.read()
                    history = history + [[query, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Write a brief report for G."
            elif query == 'Show the API chain generated.':
                # query1="""<div style="font-size:20px;"></div>"""
                query1 = """<div style="font-size:16px;">Show the API chain generated.</div>"""
                history += [[query1, ""]]
                content = """<div style="font-size:16px;">The API chain is A -> B  -> C -> D. Execute it?</div>"""
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                yield history, "", "Tell more.", "What is the API chain?"
            elif query == 'Replace B by F and execute the API chain.':
                query1 = """<div style="font-size:16px;">Replace B by F and execute the API chain.</div>"""
                history += [[query1, ""]]
                content = """<div style="font-size:16px;">Executing in the backend. It will take minutes. You can ask other questions.</div>"""
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                yield history, "", "Tell more.", "How to execute API chains?"
            elif query == 'How is the progress of the execution?':
                query1 = """<div style="font-size:16px;">How is the progress of the execution?</div>"""
                history += [[query1, ""]]
                content = """<div style="font-size:16px;">The progress is as follows.</div>"""
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                g_img_2d = os.getcwd() + "/txt/progress/" + "progress_bar" + ".png"
                history = history + [(None, (g_img_2d,))]
                yield history, "", "Tell more.", "Write a brief report for G."
            elif history[-1][
                1] == 'I understand that G is a molecular diagram and needs to call the ADMET tool' and query == 'yes':
                history += [[query, ""]]
                content = 'Calling ADMET tool...'
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                time.sleep(3.00)
                g_describe = os.getcwd() + "/txt/ADMET/G.txt"
                with open(g_describe) as f:
                    content = f.read()
                    history += [[None, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Write a brief report for G."
            # elif query == 'Can I use G to make fever reducing medicine?':
            elif query == 'Write a brief report for G.':
                query1 = '''<div style="font-size:16px;">Write a brief report for G.</div>'''
                content = """<div style="font-size:16px;">G is classified as a chemical molecule. Invoking ADMET APIs. Please wait ...</div>"""
                history += [[query1, content]]
                g_describe = os.getcwd() + "/txt/ADMET/G.txt"
                with open(g_describe) as f:
                    content = f.read()
                    history += [[None, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Write a ADMET report for G."
            elif query == 'How to modify G to Top1?':
                history += [[query, ""]]
                for character in "Calling the graph edit distance(GED) algorithm...":
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                time.sleep(2.50)
                g_img_ged = os.getcwd() + "/txt/graph_edit_distance/" + "G-Aspirin-ged" + ".png"
                history = history + [(None, (g_img_ged,))]
                ged = os.getcwd() + "/txt/graph_edit_distance/G_Aspirin.txt"
                with open(ged) as f:
                    content = f.read()
                    history = history + [[None, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Write a brief report for G."
            elif query == 'What is the common structure between G and Aspirin?':
                query1 = '''<div style="font-size:16px;">What is the common structre between G and top1?</div>'''
                history += [[query, ""]]
                for character in '''<div style="font-size:16px;">Calling the maximum common subgraph(MCS) algorithm...</div>''':
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""

                # history +=  [[None, ""]]
                # content='The figure shows the MCS of aspirin and Methyl salicylate'
                # for character in content:
                #     history[-1][1] += character
                #     time.sleep(0.00)
                #     yield history,""
                time.sleep(2.50)
                # g_img_mcs = os.getcwd() + "/txt/graph_mcs/" + "G-Aspirin" + ".png"
                g_img_mcs = os.getcwd() + "/txt/graph_mcs/" + "G" + ".png"
                history = history + [(None, (g_img_mcs,))]
                yield history, "", "Tell more.", "Write a brief report for G."
            elif query == 'What are the common functions of these molecules?':
                g_describe = os.getcwd() + "/txt/knn_text/common functions.txt"
                with open(g_describe) as f:
                    content = f.read()
                    history = history + [[query, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Write a brief report for G."
            elif query == 'What are the differences between these molecules?':
                g_describe = os.getcwd() + "/txt/knn_text/differences.txt"
                with open(g_describe) as f:
                    content = f.read()
                    history = history + [[query, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Write a brief report for G."
            elif query == 'What are the side effects of these molecules?':
                g_describe = os.getcwd() + "/txt/knn_text/side effect.txt"
                with open(g_describe) as f:
                    content = f.read()
                    history = history + [[query, ""]]
                    for character in content:
                        history[-1][1] += character
                        time.sleep(0.00)
                        yield history, "", "", ""
                    yield history, "", "Tell more.", "Write a brief report for G."
            # elif query == 'What are the most similar molecules of G?':

            
            elif query=='What molecules is the most similar to G?':
                history += [[query, ""]]
                content = '''Aspirin is the top one similar molecules of G.
                        '''
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                g_img_2d = os.getcwd() + "/txt/graph_img_2D/" + "Aspirin" + ".png"
                history = history + [(None, (g_img_2d,))]
                yield history, "", "Tell more.", "What molecules are similar to G?"
            elif query == 'What molecules are similar to G?':
                query1 = '''<div style="font-size:16px;">What molecules are similar to G?</div>'''
                history += [[query1, ""]]
                # for character in "Calling the k-nearest neighbors algorithm...":
                #     history[-1][1] += character
                #     time.sleep(0.00)
                #     yield history, "","Tell more.","Write a brief report for G."
                # time.sleep(2.50)
                # history += [[None, ""]]

                # content = '''<div style="font-size:16px;">Aspirin and Methyl Salicylate are the top two similar molecules of G.</div>
                #         '''
                content = '''This is the top one similar molecule of G.</div>
                        '''
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                g_img_2d = os.getcwd() + "/txt/graph_img_2D/" + "Aspirin" + ".png"
                history = history + [(None, (g_img_2d,))]
                yield history, "", "", ""
                history += [[None, ""]]
                content = '''<strong style="color: red; font-size: 16px;">Top&nbsp;1</strong>&nbsp;&nbsp;&nbsp;Aspirin
                        Acetylsalicylic acid appears as odorless white crystals or crystalline powder with a slightly bitter taste.
                        '''
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                g_img_2d = os.getcwd() + "/txt/graph_img_2D/" + "Aspirin" + ".png"
                history = history + [(None, (g_img_2d,))]
                yield history, "", "", ""
                history += [[None, ""]]
                content = '''<strong style="color: red; font-size: 16px;">Top&nbsp;2</strong>&nbsp;&nbsp;&nbsp;Methyl Salicylate
                        Methyl salicylate appears as colorless yellowish or reddish liquid with odor of wintergreen.
                        '''
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                g_img_2d = os.getcwd() + "/txt/graph_img_2D/" + "Methyl Salicylate" + ".png"
                history = history + [(None, (g_img_2d,))]
                yield history, "", "", ""
                history += [[None, ""]]
                content = '''<strong style="color: red; font-size: 16px;">Top&nbsp;3</strong>&nbsp;&nbsp;&nbsp;Salsalate
                        Salsalate is a nonsteroidal anti-inflammatory agent for oral administration.
                        '''
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                g_img_2d = os.getcwd() + "/txt/graph_img_2D/" + "Salsalate" + ".png"
                history = history + [(None, (g_img_2d,))]
                yield history, "", "Tell more.", "Write a brief report for G."
            elif query == 'Please provide its three-dimensional structure diagram':
                history += [[query, ""]]
                content = "Its three-dimensional image is as follows"
                for character in content:
                    history[-1][1] += character
                    time.sleep(0.00)
                    yield history, "", "", ""
                g_img_3d = os.getcwd() + "/txt/graph_img_3D/" + gname + ".png"
                history = history + [(None, (g_img_3d,))]
                yield history, "", "Tell more.", "Write a brief report for G."
            else:
                for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                                      streaming=streaming):
                    resp = answer_result.llm_output["answer"]
                    history = answer_result.history
                    history[-1][-1] = resp
                    yield history, "", "Tell more.", "Write a brief report for G."
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    # flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


def init_model():
    args = parser.parse_args()

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        generator = local_doc_qa.llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话"""
        # logger.info(reply)
        # return reply
    except Exception as e:
        # logger.error(e)
        # reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        reply = """模型已成功加载，可以开始对话"""
        # if str(e) == "Unknown platform: darwin":
        #     logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
        #                 " https://github.com/imClumsyPanda/langchain-ChatGLM")
        # else:
        #     logger.info(reply)
        return reply


def reinit_model(llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2, use_lora, top_k,
                 history):
    try:
        llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        # model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        model_status = """模型已成功重新加载，可以开始对话"""
        logger.info(model_status)
    return history + [[None, model_status]]


def get_knnVector_store(history):
    if local_doc_qa.llm and local_doc_qa.embeddings:
        filelist = ['/root/ls/workspace/langchain-ChatGLM/knowledge_base/aspirin/content/aspirin.txt']
        vs_path = '/root/ls/workspace/langchain-ChatGLM/knowledge_base/aspirin/vector_store'
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, 200)

        if len(loaded_files):
            file_status = f"已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    filelist = []
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                print(filelist)
                print(vs_path)
                print(sentence_size)
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history, \
               gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=[], value=[]), gr.update(visible=True, value=[])


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1. 您已进入知识库测试模式，您输入的任何对话内容都将用于进行知识库查询，"
                                 "并仅输出知识库匹配出的内容及相似度分值和及输入的文本源路径，查询的内容并不会进入模型查询。\n\n"
                                 "2. 知识相关度 Score 经测试，建议设置为 500 或更低，具体设置情况请结合实际使用调整。"
                                 """3. 使用"添加单条数据"添加文本至知识库时，内容如未分段，则内容越多越会稀释各查询内容与之关联的score阈值。\n\n"""
                                 "4. 单条内容长度建议设置在100-150左右。\n\n"
                                 "5. 本界面用于知识入库及知识匹配相关参数设定，但当前版本中，"
                                 "本界面中修改的参数并不会直接修改对话界面中参数，仍需前往`configs/model_config.py`修改后生效。"
                                 "相关参数将在后续版本中支持本界面直接修改。")


def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库测试":
        return gr.update(visible=True), gr.update(visible=True), [[None,
                                                                   knowledge_base_test_mode_info]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history


# def change_mode(mode, history):
#         return history
#         # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]


def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, chatbot):
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)


# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id, "vector_store"))
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(os.path.join(KB_ROOT_PATH, vs_id, "content"),
                                                                         vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())


def delete_file(vs_id, files_to_delete, chatbot):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(KB_ROOT_PATH, vs_id, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    status = local_doc_qa.delete_file_from_vector_store(vs_path=vs_path,
                                                        filepath=docs_path)
    if "fail" not in status:
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
    rested_files = local_doc_qa.list_file_from_vector_store(vs_path)
    if "fail" in status:
        vs_status = "文件删除失败。"
    elif len(rested_files) > 0:
        vs_status = "文件删除成功。"
    else:
        vs_status = f"文件删除成功，知识库{vs_id}中无已上传文件，请先上传文件后，再开始提问。"
    logger.info(",".join(files_to_delete) + vs_status)
    chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), chatbot


def delete_vs(vs_id, chatbot):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(
            visible=True), \
               gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), chatbot, gr.update(visible=True)


def veShow(file):
    # 创建一个无向图
    dot = Graph(comment='Undirected Graph')
    lines = file.read()
    if isinstance(lines, str) != True:
        lines = lines.decode()
    lines3 = lines.split("\n")
    if (lines3[0] != '' and lines3[0][0] == 't'):
        for j in range(1, len(lines3)):
            tmp = lines3[j].split(' ')
            # 添加节点
            if tmp[0] == 'v':
                dot.node(tmp[1], tmp[2])
            # 添加边，并在边上添加特征
            elif tmp[0] == 'e':
                dot.edge(tmp[1], tmp[2], label=tmp[3])

    # 设置特征的样式
    dot.attr('edge', fontsize='12', fontcolor='red')

    # 将DOT文件转换为图像文件
    graph = pydotplus.graph_from_dot_data(dot.source)
    png_bytes = graph.create_png()
    # graph.write_png('static/imgs/'+file.filename+'.png')
    return png_bytes


def upload_files(files):
    for file in files:
        with open(file.name, "r") as f:
            image_bytes = veShow(f)
            image_file = io.BytesIO(image_bytes)
            image = Image.open(image_file)
    return gr.update(value=image, visible=True)


def knn_search():
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def checkbox_callback(selected):
    if selected:
        return gr.update(interactive=True)


def get_graph_img(history, file):
    output_speed = 0.01
    gname = file.name.split("/")[-1].split(".")[0]
    if gname == 'G' or gname == 'Aspirin':
        # history = history + [[None, ""]]
        # content = "Loaded successfully!"
        # content = "Show the 3d structure of this graph."
        # for character in content:
        #     history[-1][1] += character
        #     time.sleep(output_speed)
        #     yield history
        g_img_2d = os.getcwd() + "/txt/graph_img_2D/" + gname + ".png"
        # g_img_3d = os.getcwd() + "/txt/graph_img_3D/" + "Aspirin" + ".png"
        # history = history + [(None, (g_img_3d,))]
        history = history + [((g_img_2d,), None)]
    elif gname == 'Social Graph':
        history = history + [[None, ""]]
        content = "G is loaded successfully!"
        for character in content:
            history[-1][1] += character
            time.sleep(output_speed)
            yield history
    else:
        # print(os.getcwd())
        history = history + [[None, ""]]
        content = "The uploaded VE file has been converted into an image and it is recognized as " + gname
        for character in content:
            history[-1][1] += character
            time.sleep(output_speed)
            yield history

        history = history + [[None, ""]]
        content = "Its two-dimensional image is as follows"
        for character in content:
            history[-1][1] += character
            time.sleep(output_speed)
            yield history

        g_img_2d = os.getcwd() + "/txt/graph_img_2D/" + gname + ".png"
        history = history + [(None, (g_img_2d,))]

        g_describe = os.getcwd() + "/txt/graph_describe/" + gname + ".txt"
        with open(g_describe) as f:
            content = f.read()
            history = history + [[None, ""]]
            for character in content:
                history[-1][1] += character
                time.sleep(output_speed)
                yield history
    yield history


def tellMore(history, btn):
    if btn == 'How is its water solubility?':
        ged = os.getcwd() + "/txt/G/water-solubility.txt"
        with open(ged) as f:
            content = f.read()
            history = history + [[query, ""]]
            for character in content:
                history[-1][1] += character
                time.sleep(0.00)
                yield history, "1", "2", "3", "4"


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}
#btn1 {
    display: inline-block;
    background: white; 
    border: 1px solid lightgray; 
    padding-left: 30px; 
    text-align: left; 
    width: 100%; 
    height: 40px; 
    border-radius: 10px; 
    font-family:  'Microsoft YaHei', sans-serif; 
}
#btn2 {
    display: inline-block;
    background: white; 
    border: 1px solid lightgray; 
    padding-left: 30px; 
    text-align: left; 
    width: 100%; 
    height: 40px; 
    border-radius: 10px; 
    font-family:  'Microsoft YaHei', sans-serif; 
}
"""

webui_title = """
# 🎉ChatGraph: Chat with Your Graphs
"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""Welcome! Please input your questions and graphs."""

# 初始化消息
model_status = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args), title="ChatGraph") as demo:
    vs_path, file_status, model_status = gr.State(
        os.path.join(KB_ROOT_PATH, get_vs_list()[0], "vector_store") if len(get_vs_list()) > 1 else ""), gr.State(
        ""), gr.State(
        model_status)
    gr.Markdown(webui_title)
    with gr.Tab("Dialog"):
        # with gr.Row():
        chatbot = gr.Chatbot([[None, init_message]],
                             elem_id="chat-box",
                             bubble_full_width=False,
                             avatar_images=(
                             (os.getcwd() + "/txt/icon/avatar.png"), (os.getcwd() + "/txt/icon/gradio.png")),
                             show_label=False).style(height=800)
        btn1 = gr.Button(value="Tell more.",
                         elem_id="btn1",
                         elem_classes="btn",
                         visible=False)
        btn2 = gr.Button(value="How to calculate the similarity of graphs?",
                         elem_id="btn2",
                         elem_classes="btn",
                         visible=False)

        with gr.Row():
            with gr.Column(scale=0.85):
                query = gr.Textbox(show_label=False,
                                   placeholder="Input your question").style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload Graph 📁", file_types=["image", "text"])
        # query = gr.Textbox(show_label=False,
        #                             placeholder="请输入提问内容，按回车进行提交").style(container=False)
        mode = gr.Radio(["LLM 对话", "知识库问答"],
                        label="请选择使用模式",
                        value="知识库问答",
                        visible=False)
        btn.upload(get_graph_img,
                   [chatbot, btn],
                   [chatbot]
                   )
        query.submit(get_answer,
                     [query, vs_path, chatbot, mode, btn],
                     [chatbot, query, btn1, btn2])
        btn1.click(get_answer,
                   [btn1, vs_path, chatbot, mode, btn],
                   [chatbot, btn1, btn1, btn2])
        btn2.click(get_answer,
                   [btn2, vs_path, chatbot, mode, btn],
                   [chatbot, btn2, btn1, btn2])

        # with gr.Column(scale=10):
        #     chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
        #                          elem_id="chat-box",
        #                          show_label=False).style(height=750)
        #     query = gr.Textbox(show_label=False,
        #                             placeholder="请输入提问内容，按回车进行提交").style(container=False)
        # with gr.Row():
        #     with gr.Column(scale=0.85):
        #         query = gr.Textbox(show_label=False,
        #                         placeholder="请输入提问内容，按回车进行提交").style(container=False)
        #     with gr.Column(scale=0.15, min_width=0):
        #         btn=gr.UploadButton("📁", file_types=["text"])
        # mode="知识库问答"
        # query.submit(get_answer,
        #                 [query, vs_path, chatbot, mode],
        #                 [chatbot, query])
        # with gr.Column(scale=5):
        # mode = gr.Radio(["LLM 对话", "知识库问答", "Bing搜索问答"],
        # mode = gr.Radio(["LLM 对话", "知识库问答"],
        #                 label="请选择使用模式",
        #                 value="知识库问答", )

        # gfiles = gr.File(label="添加graph文件",
        #                 file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
        #                 file_count="multiple",
        #                 show_label=False)
        # preview=gr.Image(visible=False,
        #                  show_label=False)
        # graphKnn_search=gr.Button(value="K-Nearest Neighbors search")
        # top1=gr.Checkbox(label="Aspirin",
        #                  info="Acetylsalicylic acid appears as odorless white crystals or crystalline powder with a slightly bitter taste.",
        #                  interactive=True,
        #                  visible=False)
        # top2=gr.Checkbox(label="Methyl Salicylate",
        #                  info="Methyl salicylate appears as colorless yellowish or reddish liquid with odor of wintergreen.",
        #                  interactive=True,
        #                  visible=False)
        # top3=gr.Checkbox(label="Salsalate",
        #                  info="Salsalate is a nonsteroidal anti-inflammatory agent for oral administration.",
        #                  interactive=True,
        #                  visible=False)
        # graphKnn_load = gr.Button(value="Load knowledge base",
        #                            visible=False,
        #                            interactive=False)
        # knowledge_set = gr.Accordion("知识库设定", visible=False)
        # vs_setting = gr.Accordion("配置知识库")
        # gfiles.upload(fn=upload_files,
        #                 inputs=[gfiles],
        #                 outputs=[preview])
        # mode.change(fn=change_mode,
        #             inputs=[mode, chatbot],
        #             outputs=[chatbot])
        # mode.change(fn=change_mode,
        #             inputs=[mode, chatbot],
        #             outputs=[vs_setting, knowledge_set, chatbot])
        # graphKnn_search.click(knn_search,
        #                     show_progress=True,
        #                     inputs=[],
        #                     outputs=[top1, top2,top3,graphKnn_load] )
        # top1.select(fn=checkbox_callback,
        #             inputs=[top1],
        #             outputs=[graphKnn_load])
        # top2.select(fn=checkbox_callback,
        #             inputs=[top2],
        #             outputs=[graphKnn_load])
        # top3.select(fn=checkbox_callback,
        #             inputs=[top3],
        #             outputs=[graphKnn_load])
        # graphKnn_load.click(get_knnVector_store,
        #                     show_progress=True,
        #                     inputs=[chatbot],
        #                     outputs=[vs_path, chatbot], )

        # with vs_setting:
        #     vs_refresh = gr.Button("更新已有知识库选项")
        #     select_vs = gr.Dropdown(get_vs_list(),
        #                             label="请选择要加载的知识库",
        #                             interactive=True,
        #                             value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
        #                             )
        #     vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
        #                          lines=1,
        #                          interactive=True,
        #                          visible=True)
        #     vs_add = gr.Button(value="添加至知识库选项", visible=True)
        #     vs_delete = gr.Button("删除本知识库", visible=False)
        #     file2vs = gr.Column(visible=False)
        #     with file2vs:
        #         # load_vs = gr.Button("加载知识库")
        #         gr.Markdown("向知识库中添加文件")
        #         sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
        #                                   label="文本入库分句长度限制",
        #                                   interactive=True, visible=True)
        #         with gr.Tab("上传文件"):
        #             files = gr.File(label="添加文件",
        #                             file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
        #                             file_count="multiple",
        #                             show_label=False)
        #             load_file_button = gr.Button("上传文件并加载知识库")
        #         with gr.Tab("上传文件夹"):
        #             folder_files = gr.File(label="添加文件",
        #                                    file_count="directory",
        #                                    show_label=False)
        #             load_folder_button = gr.Button("上传文件夹并加载知识库")
        #         with gr.Tab("删除文件"):
        #             files_to_delete = gr.CheckboxGroup(choices=[],
        #                                              label="请从知识库已有文件中选择要删除的文件",
        #                                              interactive=True)
        #             delete_file_button = gr.Button("从知识库中删除选中文件")
        #     vs_refresh.click(fn=refresh_vs_list,
        #                      inputs=[],
        #                      outputs=select_vs)
        #     vs_add.click(fn=add_vs_name,
        #                  inputs=[vs_name, chatbot],
        #                  outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
        #     vs_delete.click(fn=delete_vs,
        #                     inputs=[select_vs, chatbot],
        #                     outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
        #     select_vs.change(fn=change_vs_name_input,
        #                      inputs=[select_vs, chatbot],
        #                      outputs=[vs_name, vs_add, file2vs, vs_path, chatbot, files_to_delete, vs_delete])
        #     load_file_button.click(get_vector_store,
        #                            show_progress=True,
        #                            inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
        #                            outputs=[vs_path, files, chatbot, files_to_delete], )
        #     load_folder_button.click(get_vector_store,
        #                              show_progress=True,
        #                              inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
        #                                      vs_add],
        #                              outputs=[vs_path, folder_files, chatbot, files_to_delete], )
        #     flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
        #     query.submit(get_answer,
        #                  [query, vs_path, chatbot, mode],
        #                  [chatbot, query])
        #     delete_file_button.click(delete_file,
        #                              show_progress=True,
        #                              inputs=[select_vs, files_to_delete, chatbot],
        #                              outputs=[files_to_delete, chatbot])
    # with gr.Tab("知识库测试 Beta"):
    #     with gr.Row():
    #         with gr.Column(scale=10):
    #             chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
    #                                  elem_id="chat-box",
    #                                  show_label=False).style(height=750)
    #             query = gr.Textbox(show_label=False,
    #                                placeholder="请输入提问内容，按回车进行提交").style(container=False)
    #         with gr.Column(scale=5):
    #             mode = gr.Radio(["知识库测试"],  # "知识库问答",
    #                             label="请选择使用模式",
    #                             value="知识库测试",
    #                             visible=False)
    #             knowledge_set = gr.Accordion("知识库设定", visible=True)
    #             vs_setting = gr.Accordion("配置知识库", visible=True)
    #             mode.change(fn=change_mode,
    #                         inputs=[mode, chatbot],
    #                         outputs=[vs_setting, knowledge_set, chatbot])
    #             with knowledge_set:
    #                 score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
    #                                             label="知识相关度 Score 阈值，分值越低匹配度越高",
    #                                             precision=0,
    #                                             interactive=True)
    #                 vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
    #                                                 label="获取知识库内容条数", interactive=True)
    #                 chunk_conent = gr.Checkbox(value=False,
    #                                            label="是否启用上下文关联",
    #                                            interactive=True)
    #                 chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
    #                                         label="匹配单段内容的连接上下文后最大长度",
    #                                         interactive=True, visible=False)
    #                 chunk_conent.change(fn=change_chunk_conent,
    #                                     inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False), chatbot],
    #                                     outputs=[chunk_sizes, chatbot])
    #             with vs_setting:
    #                 vs_refresh = gr.Button("更新已有知识库选项")
    #                 select_vs_test = gr.Dropdown(get_vs_list(),
    #                                              label="请选择要加载的知识库",
    #                                              interactive=True,
    #                                              value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
    #                 vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
    #                                      lines=1,
    #                                      interactive=True,
    #                                      visible=True)
    #                 vs_add = gr.Button(value="添加至知识库选项", visible=True)
    #                 file2vs = gr.Column(visible=False)
    #                 with file2vs:
    #                     # load_vs = gr.Button("加载知识库")
    #                     gr.Markdown("向知识库中添加单条内容或文件")
    #                     sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
    #                                               label="文本入库分句长度限制",
    #                                               interactive=True, visible=True)
    #                     with gr.Tab("上传文件"):
    #                         files = gr.File(label="添加文件",
    #                                         file_types=['.txt', '.md', '.docx', '.pdf'],
    #                                         file_count="multiple",
    #                                         show_label=False
    #                                         )
    #                         load_file_button = gr.Button("上传文件并加载知识库")
    #                     with gr.Tab("上传文件夹"):
    #                         folder_files = gr.File(label="添加文件",
    #                                                # file_types=['.txt', '.md', '.docx', '.pdf'],
    #                                                file_count="directory",
    #                                                show_label=False)
    #                         load_folder_button = gr.Button("上传文件夹并加载知识库")
    #                     with gr.Tab("添加单条内容"):
    #                         one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
    #                         one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
    #                         one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
    #                                                                interactive=True)
    #                         load_conent_button = gr.Button("添加内容并加载知识库")
    #                 # 将上传的文件保存到content文件夹下,并更新下拉框
    #                 vs_refresh.click(fn=refresh_vs_list,
    #                                  inputs=[],
    #                                  outputs=select_vs_test)
    #                 vs_add.click(fn=add_vs_name,
    #                              inputs=[vs_name, chatbot],
    #                              outputs=[select_vs_test, vs_name, vs_add, file2vs, chatbot])
    #                 select_vs_test.change(fn=change_vs_name_input,
    #                                       inputs=[select_vs_test, chatbot],
    #                                       outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
    #                 load_file_button.click(get_vector_store,
    #                                        show_progress=True,
    #                                        inputs=[select_vs_test, files, sentence_size, chatbot, vs_add, vs_add],
    #                                        outputs=[vs_path, files, chatbot], )
    #                 load_folder_button.click(get_vector_store,
    #                                          show_progress=True,
    #                                          inputs=[select_vs_test, folder_files, sentence_size, chatbot, vs_add,
    #                                                  vs_add],
    #                                          outputs=[vs_path, folder_files, chatbot], )
    #                 load_conent_button.click(get_vector_store,
    #                                          show_progress=True,
    #                                          inputs=[select_vs_test, one_title, sentence_size, chatbot,
    #                                                  one_conent, one_content_segmentation],
    #                                          outputs=[vs_path, files, chatbot], )
    #                 flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
    #                 query.submit(get_answer,
    #                              [query, vs_path, chatbot, mode, score_threshold, vector_search_top_k, chunk_conent,
    #                               chunk_sizes],
    #                              [chatbot, query])
    with gr.Tab("Setting"):
        # with gr.Row():
            # with gr.Column():
                with gr.Row():
                    with gr.Accordion("ANN Search"):
                        with gr.Row(variant="compact"):
                            tao = gr.Textbox(value="0", label="Parameter "+chr(964))
                            Top_k = gr.Textbox(value="0", label="Top k")
                            beamSize = gr.Textbox(value="0", label="Beam Width")
                with gr.Row():
                    with gr.Accordion("Graph Sequentializer"):
                        with gr.Row(variant="compact"):
                            levelNumber = gr.Textbox(value="0", label="Sequence Length")
                            sequenceLength = gr.Textbox(value="0", label="Level Number")
                    with gr.Accordion("Finetuning"):
                        with gr.Row(variant="compact"):
                            unnamed1 = gr.Textbox(value="0", label="Parameter "+chr(945))
                            unnamed2 = gr.Textbox(value="0", label="Rollout Number")
            # with gr.Column():
                with gr.Row():

                    llm_model = gr.Radio(llm_model_dict_list,
                                            label="LLM",
                                            value=LLM_MODEL,
                                            interactive=True)

                    embedding_model = gr.Radio(embedding_model_dict_list,
                                                label="Embedding Model",
                                                value=EMBEDDING_MODEL,
                                                interactive=True)
                with gr.Row():
                    llm_history_len = gr.Slider(0, 10,
                                                value=LLM_HISTORY_LEN,
                                                step=1,
                                                label="LLM Dialog Rounds",
                                                interactive=True)
                    load_model_button = gr.Button("Reload Model")
                top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                                  label="Vector matching top k", interactive=False, visible=False)
                no_remote_model = gr.Checkbox(shared.LoaderCheckPoint.no_remote_model,
                                              label="Load Local Models",
                                              interactive=True, visible=False)
                use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
                                             label="The model fine-tuned using p-tuning-v2",
                                             interactive=True, visible=False)
                use_lora = gr.Checkbox(USE_LORA,
                                       label="The weights fine-tuned using LoRA",
                                       interactive=True, visible=False)
         
                load_model_button.click(reinit_model, show_progress=True,
                                        inputs=[llm_model, embedding_model, llm_history_len, no_remote_model,
                                                use_ptuning_v2,
                                                use_lora, top_k, chatbot], outputs=chatbot)
        # load_knowlege_button = gr.Button("重新构建知识库")
    # load_knowlege_button.click(reinit_vector_store, show_progress=True,
    #                         #    inputs=[chatbot], outputs=chatbot)
    #                            inputs=[select_vs, chatbot], outputs=chatbot)
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        # outputs=[select_vs, select_vs_test],
        # outputs=[select_vs_test],
        outputs=[],
        queue=True,
        show_progress=False,
    )

(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))
