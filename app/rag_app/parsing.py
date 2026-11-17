from fastapi import HTTPException, UploadFile


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
    return [p.strip() for p in content.split("\n\n") if p.strip()]
