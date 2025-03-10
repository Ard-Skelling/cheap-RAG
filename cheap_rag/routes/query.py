import os
from fastapi import APIRouter, HTTPException


router = APIRouter()


# local module
from routes.schema import DataSearchEngineRequest, APIResponse
from modules.workflow.workflow_paper.query_workflow.pipe import QueryWorkflow


QUERY_WORKFLOW = QueryWorkflow()


@router.post('/search')
async def search(request: DataSearchEngineRequest):
    try:
        result = await QUERY_WORKFLOW.search(
            query=request.query,
            domain=request.domain,
            topk=request.topk,
            output_fields=request.output_fields,
            threshold=request.threshold
        )
        return APIResponse(
            status=200,
            result=result
        ).model_dump()
    except Exception as err:
        return APIResponse(
            status=10001,
            message=repr(err)
        ).model_dump()
