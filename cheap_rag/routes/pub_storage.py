import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse


router = APIRouter()


# local module
from modules.storage.minio_storage import MINIO_STORAGE


@router.get('/image')
async def get_image(domain:str, file:str, image:str):
    obj_name = f'{domain}/{file}.pdf/images/{image}.jpg'
    try:
        obj_bytes = io.BytesIO(MINIO_STORAGE.get_object(obj_name, 'file-ocr'))
        return StreamingResponse(obj_bytes, media_type='image/jpeg')
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found or error: {str(e)}")


