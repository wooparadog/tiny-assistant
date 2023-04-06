from os import path
from typing import List, cast
from uuid import uuid4
from zipfile import ZipFile

import gradio as gr

from langchain import OpenAI

from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR
from llama_index.readers.file.base_parser import ImageParserOutput
from llama_index import (
    Document,
    GPTSimpleVectorIndex,
    LLMPredictor,
    ServiceContext,
)


def zip_to_json(file_obj):
    files = []
    with ZipFile(file_obj.name) as zfile:
        for zinfo in zfile.infolist():
            files.append(
                {
                    "name": zinfo.filename,
                    "file_size": zinfo.file_size,
                    "compressed_size": zinfo.compress_size,
                }
            )
    return files


async def converse(user_id, msg, history):
    history.append([msg, None])
    index = get_index(user_id)
    query = await index.aquery(msg)
    history[-1][1] = query.response

    return (history, "")


def get_index(user_id) -> GPTSimpleVectorIndex:
    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo-0301")
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    model_file = path.join("data", f"{user_id}.json")
    if path.exists(model_file):
        print(f"getting index from disk: {user_id}")
        return cast(
            GPTSimpleVectorIndex,
            GPTSimpleVectorIndex.load_from_disk(
                model_file, service_context=service_context
            ),
        )
    else:
        print(f"creating new index: {user_id}")
        return cast(
            GPTSimpleVectorIndex,
            GPTSimpleVectorIndex.from_documents([], service_context=service_context),
        )


def save_index(index, user_id):
    model_file = path.join("data", f"{user_id}.json")
    index.save_to_disk(model_file)


def parse_document(file_obj) -> List[Document]:
    _, file_extension = path.splitext(file_obj.name)
    parser = DEFAULT_FILE_EXTRACTOR.get(file_extension.lower())
    if not parser:
        raise ValueError("不支持的文件")

    if not parser.parser_config_set:
        parser.init_parser()
    content = parser.parse_file(file_obj.name)
    ret = []
    if isinstance(content, List):
        ret.append(Document("\n".join(content)))
    elif isinstance(content, ImageParserOutput):
        pass
    else:
        ret.append(Document(content))
    return ret


def handle_process_file(user_id, files):
    index = get_index(user_id)
    for file in files:
        print(f"adding {file.name} to index")
        for doc in parse_document(file):
            index.insert(doc)
    save_index(index, user_id)


title = """<h1 align="center">小助手demo</h1>"""


with gr.Blocks() as demo:
    gr.HTML(title)
    user_id = gr.State(value=uuid4().hex)

    def set_user_id(uid):
        return uid, uid

    with gr.Row():
        with gr.Column(scale=2):
            user_id_textbox = gr.Textbox(
                label="用户id", info="你的唯一标识符, 最好使用 uuid, 请记住这个字符串", value=user_id.value
            )
            files = gr.File(file_count="multiple", info="上传训练文件")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="聊天框", info="")
            msg_txt_box = gr.Textbox(
                label="输入框",
                info="输入你的内容，按[Enter]发送。也可以什么都不填写生成随机数据。对话一般不能太长，否则就复读机了，建议清除数据。",
            )
            clear = gr.Button("清除聊天")

    user_id_textbox.submit(set_user_id, [user_id_textbox], [user_id_textbox, user_id])
    files.upload(handle_process_file, [user_id, files], None)
    msg_txt_box.submit(
        converse, [user_id, msg_txt_box, chatbot], [chatbot, msg_txt_box]
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
