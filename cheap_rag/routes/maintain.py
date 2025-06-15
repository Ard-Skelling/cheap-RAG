from os import getenv
from fastapi import APIRouter, HTTPException

# Local module
from routes.schema import (
    VecDataCreateEngineRequest,
    VecDataDeleteEngineRequest,
    DataDeleteEngineRequest,
    FileNameSearchEngineRequest,
    APIResponse
)
from modules.workflow.workflow_paper.db_opration import PAPER_KNOWLEDGE


router = APIRouter()


@router.post('/create_paper_collection')
async def create_paper_collection(request: VecDataCreateEngineRequest):
    collections = request.collections
    token = request.token
    if token != getenv('LOCAL_TOKEN'):
        raise HTTPException(status_code=401, detail="Invalid token")
    await PAPER_KNOWLEDGE.create_db(collections)
    return APIResponse(status=200, message=f"Create paper collection successfully: {collections}")


@router.post('/delete_paper_collection')
async def delete_paper_collection(request: VecDataDeleteEngineRequest):
    collections = request.collections
    token = request.token
    if token != getenv('LOCAL_TOKEN'):
        raise HTTPException(status_code=401, detail="Invalid token")
    await PAPER_KNOWLEDGE.delete_db(collections)
    return APIResponse(status=200, message=f"Delete paper collection successfully: {collections}")


@router.post('/delete_paper')
async def delete_paper(request: DataDeleteEngineRequest):
    domain = request.domain
    file_names = request.file_name
    await PAPER_KNOWLEDGE.delete_file(domain, file_names)
    return APIResponse(status=200, message=f"Delete paper successfully: {file_names}")


@router.post('/list_paper')
async def list_paper(request: FileNameSearchEngineRequest):
    domain = request.domain
    file_names = await PAPER_KNOWLEDGE.list_files(domain)
    return APIResponse(status=200, result=file_names)


