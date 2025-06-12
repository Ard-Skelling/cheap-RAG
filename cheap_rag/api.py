import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# local module
from routes.insert import router as insert_router
from routes.query import router as query_router
from routes.tool import router as tool_router
from routes.pub_storage import router as storage_router
from routes.maintain import router as maintain_router


# API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix='/api/v1/query', tags=['query'])
app.include_router(tool_router, prefix='/api/v1/tool', tags=['tool'])
app.include_router(insert_router, prefix='/api/v1/insert', tags=['insert'])
app.include_router(storage_router, prefix='/api/v1/storage', tags=['storage'])
app.include_router(maintain_router, prefix='/api/v1/maintain', tags=['maintain'])


@app.get("/api/health")
async def health():
    return {"status": 200, "message": "OK"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, workers=1)