import os
from fastapi import APIRouter, HTTPException
from concurrent.futures import ProcessPoolExecutor


router = APIRouter()


# local module
from routes.schema import (
    VecDataCreateEngineRequest,
    VecDataDeleteEngineRequest,
    DataDeleteEngineRequest,
    ObjStorageEngineRequest
)


@router.post('/create_domain')
async def create_domain(request: VecDataCreateEngineRequest):
    # TODO: Create data collection
    raise NotImplementedError


@router.post('/delete_domain')
async def delete_domain(request: VecDataDeleteEngineRequest):
    # TODO: Delete data collection
    raise NotImplementedError


@router.post('/delete_file')
async def delete_file(request: DataDeleteEngineRequest):
    # TODO: Delete file(s)
    raise NotImplementedError


@router.post('/object_storage')
async def object_storage(request: ObjStorageEngineRequest):
    # TODO: Upload/download object
    raise NotImplementedError
