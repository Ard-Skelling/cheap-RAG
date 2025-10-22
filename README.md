# cheap-RAG
A cheap but strong Retrieval-Augmented Generation (RAG) framework.

## Features
- Lightweight, cost-effective, and easy to deploy
- Supports document collection management, insertion, querying, and maintenance
- Modular API design for flexible integration

## Workflow
### Collection Level Operation
- Create and delete data collections (Hybrid Storage)
- List and delete documents in collections
- Collection maintenance

### Data Level Operation
- Insert documents (PDF in base64 string)
- Query documents with semantic search

![data_workflow](cheap_rag/assets/images/RAG_workflow.jpg)

---


## Pre-requirements
Before you begin, make sure you have the following prerequisites set up.

### 1. Environment Variables
Set the access and secret keys for MinIO:
```bash
export MINIO_ACCESS_KEY="xxx"
export MINIO_SECRET_KEY="xxx"
```

### 2. Prepare Directories
Create the necessary directories for Elasticsearch and Milvus, and set permissions:
```bash
mkdir -p /database/es/data
chmod -R 770 /database/es/data
cp milvus-config.yaml /database/milvus-config.yaml
```

### 3. Start Services
Use Docker Compose to start the required services in detached mode:
```bash
docker compose up -d
```


## Installation & Run
1. **Install dependencies**
   ```bash
   poetry install --extras cpu
   # Use GPU
   # poetry install --extras gpu
   ```
2. **Download model weights**
  ```bash
  bash download_model.sh
  ```
3. **Modify config**
  Modify the config in `cheap_rag/configs/config.py` and `cheap_rag/modules/workflow/workflow_paper/config.py`
4. **Start the API server**
   ```bash
   eval $(poetry env activate)
   python cheap_rag/api.py
   ```
   The server will run at `http://localhost:8002` by default.


---

## API Endpoints

### Health Check
- **GET** `/api/health`
  - **Response:** `{ "status": 200, "message": "OK" }`

### Document Insertion
- **POST** `/api/v1/insert/insert_pdf`
  - **Body:**
    ```json
    {
      "csid": "string",           // Client UUID
      "file_bs64": "string",      // Base64 encoded PDF
      "domain": "string",         // Collection name
      "file_name": "string",      // File name
      "file_meta": { ... },        // (Optional) File metadata
      "ocr_api": "string",        // (Optional) OCR API endpoint
      "emb_api": "string"         // (Optional) Embedding API endpoint
    }
    ```
  - **Response:**
    ```json
    { "status": 200, "result": ..., "message": "SUCCESS" }
    ```

### Document Query
- **POST** `/api/v1/query/search`
  - **Body:**
    ```json
    {
      "csid": "string",
      "query": "string",           // Query text
      "domain": "string",          // Collection name
      "threshold": 0.01,            // (Optional) Similarity threshold
      "topk": 10,                   // (Optional) Number of results
      "output_fields": ["text", "file_name", ...] // (Optional) Fields to return
    }
    ```
  - **Response:**
    ```json
    { "status": 200, "result": [...], "message": "SUCCESS" }
    ```

### Embedding Service
- **POST** `/api/v1/tool/embedding`
  - **Body:**
    ```json
    {
      "csid": "string",
      "contents": ["string", ...], // Texts to embed
      "token": "string"            // Embedding service token
    }
    ```
  - **Response:**
    ```json
    { "status": 200, "result": [...], "message": "SUCCESS" }
    ```

### Image Retrieval
- **GET** `/api/v1/storage/image?domain=xxx&file=xxx&image=xxx`
  - **Query Params:**
    - `domain`: Collection name
    - `file`: File name
    - `image`: Image name (without .jpg)
  - **Response:** JPEG image stream

### Collection Maintenance
- **POST** `/api/v1/maintain/create_paper_collection`
  - **Body:**
    ```json
    {
      "csid": "string",
      "collections": ["string", ...],
      "token": "string" // Admin token
    }
    ```
- **POST** `/api/v1/maintain/delete_paper_collection`
  - **Body:** Same as above
- **POST** `/api/v1/maintain/delete_paper`
  - **Body:**
    ```json
    {
      "csid": "string",
      "domain": "string",
      "file_name": ["string", ...]
    }
    ```
- **POST** `/api/v1/maintain/list_paper`
  - **Body:**
    ```json
    {
      "csid": "string",
      "domain": "string"
    }
    ```

---

## Response Format
All API responses (except image stream) follow:
```json
{
  "status": 200,
  "result": ...,   // Data or list
  "message": "SUCCESS",
  "time_cost": 0.123 // (Optional) Time cost in seconds
}
```

---