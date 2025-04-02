import uvicorn
from fastapi import FastAPI


# local module
from routes.insert import router as insert_router
from routes.query import router as query_router
from routes.tool import router as tool_router


# API
app = FastAPI()

app.include_router(query_router, prefix='/api/v1/query', tags=['query'])
app.include_router(tool_router, prefix='/api/v1/tool', tags=['tool'])
app.include_router(insert_router, prefix='/api/v1/insert', tags=['insert'])


@app.get("/test_get")
async def test_get():
    return {"message": "Welcome to the API"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, workers=1)