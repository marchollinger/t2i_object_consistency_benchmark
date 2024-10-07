import base64
import io
import os
from PIL import Image

import numpy as np


def get_base64_url(path):
    if not isinstance(path, io.BytesIO):
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = path.getvalue()
    image = base64.b64encode(data)
    prefix = "data:image/png;base64,"
    return prefix + image.decode()


def get_empty_img():
    image = Image.fromarray(np.zeros((768,768)))
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    prefix = "data:image/png;base64,"
    return prefix + img_str.decode()

def get_images_from_folder(path: str) -> dict[str, str]:
    out = {}
    with os.scandir(path) as entries:
        for entry in entries:
            url = get_base64_url(entry.path)
            name = entry.name.split(sep=".", maxsplit=1)[0]
            out[name] = url

    return out

if __name__ == "__main__":
    # print(get_empty_img())
    print(get_base64_url("langchain/final/t1/scenes/scn_001/zm_4/back.png"))