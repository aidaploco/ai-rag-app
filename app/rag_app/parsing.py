from fastapi import HTTPException, UploadFile


def split_pages(text: str) -> list[str]:
    """Split text into paragraph "pages": blank-line separated, stripped, empties dropped."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


async def read_pages(file: UploadFile) -> list[str]:
    """Decode an uploaded text file and split it into paragraph "pages".

    This is the single place uploads are turned into text, so supporting new document types
    means extending this function rather than every route.
    """
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="Only UTF-8 text files are supported") from None
    return split_pages(content)
