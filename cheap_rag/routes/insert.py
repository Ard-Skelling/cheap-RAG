import os
from fastapi import APIRouter, HTTPException
from concurrent.futures import ProcessPoolExecutor


router = APIRouter()


# local module
from routes.schema import DataInsertEngineRequest, APIResponse
from modules.workflow.workflow_paper.config import WORKER_CONFIG
from modules.workflow.workflow_paper.data_cls import TaskMeta
from modules.workflow.workflow_paper.insert_workflow.pipe import InsertWorkflow
from modules.task_manager import CoroTaskManager


WORKFLOW = InsertWorkflow()
TASK_MANAGER = CoroTaskManager()
POOL = ProcessPoolExecutor(WORKER_CONFIG.num_workers)


@router.post('/insert_pdf')
async def insert(request: DataInsertEngineRequest):
    try:
        task_meta = TaskMeta(
            domain=request.domain,
            file_name=request.file_name,
            csid=request.csid,
            init_step='to_ocr',
            result=request.file_bs64
        )
        result = await WORKFLOW.submit(TASK_MANAGER, POOL, task_meta)
        return APIResponse(
            status=200,
            result=result
        ).model_dump()
    except Exception as err:
        return APIResponse(
            status=10001,
            message=repr(err)
        ).model_dump()
