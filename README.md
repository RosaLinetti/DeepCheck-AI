# DeepCheck-AI

DeepCheck-AI is an AI-powered semantic plagiarism detection API designed to identify intelligently rephrased content. It goes beyond simple string matching by using NLP and embeddings (SBERT Transformers) to detect semantic similarities across chunks of text.

---

## 1. Running the Application

### Prerequisites
- Python 3.9+
- A virtual environment (recommended)

### Steps
1. **Clone the repository**:
   ```bash
   cd DeepCheck-AI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server**:
   ```bash
   uvicorn app.api.main:app --reload
   ```
   *The server runs on `http://127.0.0.1:8000`. You can test endpoints via Swagger UI at `http://127.0.0.1:8000/docs`.*

---

## 2. Endpoints

### `POST /document/analyze/upload`
**Purpose**: Full document upload and semantic analysis pipeline. Parses PDF, DOCX, or TXT files and checks them against each other.

**Example Request (Bash)**:
```bash
curl -X POST "http://127.0.0.1:8000/document/analyze/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "source_file=@original.pdf" \        # also accepts .docx or .txt
  -F "suspicious_file=@suspect.pdf" \      # also accepts .docx or .txt
  -F "chunk_strategy=sentence"
```

**Example Output**:
```json
{
  "chunk_strategy": "sentence",
  "total_suspicious_chunks": 120,
  "overall_similarity": 0.3542,
  "max_similarity": 0.9841,
  "chunk_matches": [
    {
      "suspicious_chunk_index": 0,
      "suspicious_chunk_text": "This is a potentially plagiarized sentence.",
      "best_match_source_index": 5,
      "best_match_source_text": "This is a sentence that might be plagiarized.",
      "similarity_score": 0.8921,
      "verdict": "plagiarised",
      "confidence": 0.94
    }
  ]
}
```

### `POST /document/analyze`
**Purpose**: Direct text-based chunk analysis (no file upload required). 

**Example Request (Bash)**:
```bash
curl -X POST "http://127.0.0.1:8000/document/analyze" \
  -H "Content-Type: application/json" \
  -d '{
        "source_document": "The core logic revolves around breaking documents into smaller segments.",
        "suspicious_document": "The main idea is splitting files into small segments.",
        "chunk_strategy": "sentence"
      }'
```

### `POST /analyze`
**Purpose**: Legacy endpoint for basic text-to-text semantic similarity analysis.

**Example Request (Bash)**:
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
        "text1": "AI is transforming the world.",
        "text2": "Artificial Intelligence is changing our lives."
      }'
```

**Example Output**:
```json
{
  "similarity_score": 85.4,
  "message": "Semantic similarity analysis completed."
}
```

---

## 3. Implementation Logic

DeepCheck-AI relies on several interconnected services and machine learning models to provide accurate, multi-faceted plagiarism detection.

### Core Components & Models
- **SBERT Embedder**: Uses the `all-MiniLM-L6-v2` model (via `sentence-transformers`) to generate rich vector embeddings for textual chunks. 
- **Plagiarism Classifier**: A lightweight Scikit-Learn `LogisticRegression` model, trained on hand-crafted features, to predict categorical verdicts ("original", "suspicious", "plagiarised") along with a confidence score.
- **Document Parsers**: Custom extraction logic using `PyPDF2`, `python-docx`, and direct UTF-8/Latin-1 decoding for plain-text files. Memory is managed efficiently by chunk-reading uploads up to 50MB.

### How Similarity is Calculated

1. **Chunking**: Documents are split into segments.
   - **Sentence**: Splits based on punctuation boundaries.
   - **Sliding Window**: Splits into fixed token lengths (e.g., 30 tokens) with overlaps (e.g., 10 tokens) to preserve context.
2. **Vectorization**: Each chunk is passed through the `EmbeddingService` to produce numerical vectors.
3. **Similarity Matrix**: The system computes a base **Cosine Similarity** matrix comparing every suspicious chunk against every source chunk using `sklearn.metrics.pairwise`.
4. **Multi-Feature Scoring**: For the best matches, the system computes a final `similarity_score` which is a weighted combination of:
   - **Cosine Similarity** (Weight: 0.6) - Semantic meaning.
   - **Length Ratio** (Weight: 0.2) - Compares the size of the chunks.
   - **Lexical Overlap** (Weight: 0.2) - Jaccard index of shared words.
5. **Classification Verdict**: These three features (cosine, length ratio, lexical overlap) are passed into the `LogisticRegression` classifier, which returns a categorical `verdict` and a `confidence` level.
6. **Aggregations**: Computes metrics like `overall_similarity` (mean of all final scores) and `max_similarity`.
