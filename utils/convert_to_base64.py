import base64
import io


def get_base64_url(path_or_buffer: io.BytesIO | str) -> str:
    """Encode an image in base64.

    Args:
        path_or_buffer: The path of a png image or a BytesIO object containing a png image.

    Returns:
        The image encoded as a base64 url.

    """
    if not isinstance(path_or_buffer, io.BytesIO):
        with open(path_or_buffer, "rb") as f:
            data = f.read()
    else:
        data = path_or_buffer.getvalue()
    image = base64.b64encode(data)
    prefix = "data:image/png;base64,"
    return prefix + image.decode()
