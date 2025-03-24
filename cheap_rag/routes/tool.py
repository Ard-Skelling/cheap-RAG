import os
from fastapi import APIRouter, HTTPException


router = APIRouter()


# local module
from routes.schema import EmbeddingRequest, APIResponse
from utils.tool_calling.local_inferring.torch_inference import LocalEmbedding


@router.post('/embedding')
async def embedding(request: EmbeddingRequest):
    try:
        model = LocalEmbedding()
        result = await model.a_embedding(request.contents)
        return APIResponse(
            status=200,
            result=result
        ).model_dump()
    except Exception as err:
        return HTTPException(
            status_code=400,
            detail=repr(err)
        )