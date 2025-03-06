import uvicorn
from fastapi import FastAPI


# local module
from routes.query import router as query_router


# API
app = FastAPI()

app.include_router(query_router, prefix='/api/v1/query', tags=['query'])


@app.get("/test_get")
async def test_get():
    return {"message": "Welcome to the API"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, workers=1)