
---
title: Mental Health Chatbot Backend
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---
# RAG-Based Mental Health Chatbot 🧠💬

## Table of Contents
1. [Project Overview](#project-overview)
2. [NLP Modules](#nlp-modules)
3. [Development Journey](#development-journey)
4. [RAG Architecture](#rag-architecture)
5. [Unit Testing](#unit-testing)
6. [Containerization](#containerization)
7. [Connecting to Axiom](#connecting-to-axiom)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Deployed on Hugging Face](#deployed-on-hugging-face)
10. [Project Structure](#project-structure)
11. [Quick Start Guide](#quick-start-guide)
12. [Conclusion](#conclusion)

---

## Project Overview

The **RAG-Based Mental Health Chatbot** is an intelligent conversational system designed to provide empathetic, context-aware mental health support. By combining cutting-edge Natural Language Processing (NLP) techniques with Retrieval-Augmented Generation (RAG), this chatbot delivers personalized mental health guidance while maintaining sensitivity to emotional nuances.

### Why This Project Matters

Mental health support is increasingly critical in today's fast-paced world. This chatbot addresses the gap in accessible mental health resources by:
- **24/7 Availability**: Offering support whenever users need it
- **Personalized Responses**: Understanding user emotions and intents to tailor responses
- **Knowledge-Grounded**: Leveraging a curated knowledge base of mental health strategies and coping mechanisms
- **Multilingual Support**: Breaking language barriers to reach a global audience
- **Empathetic Engagement**: Recognizing emotional states to respond with appropriate tone and content

---

## NLP Modules

The chatbot's intelligence is powered by three core NLP modules that work in harmony:

### 1. Language Detection Module
**Purpose**: Identifies the language in which the user is communicating.

**Key Features**:
- Supports 40+ languages globally
- Returns confidence scores for robust decision-making
- Falls back gracefully with low-confidence flagging
- Returns top-5 language predictions

**Contribution to the System**:
- Enables multilingual support by determining if translation is needed
- Feeds language code to the translation pipeline
- Ensures RAG retrieval uses the correct context

### 2. Emotion Classification Module
**Purpose**: Detects and classifies emotional states from user input.

**Emotions Detected**: 
- Sadness, Joy, Love, Anger, Fear, Surprise

**Key Features**:
- Fine-tuned DistilBERT model trained on GoEmotions/dair-ai emotion dataset
- Confidence scores for each emotion
- Crisis flag detection for high-intensity negative emotions (sadness/fear ≥ 90% confidence)
- Generates tone hints for RAG response generation

**Contribution to the System**:
- Personalizes chatbot tone (empathetic, celebratory, reassuring, etc.)
- Triggers crisis protocols when detecting severe emotional distress
- Guides RAG system to select appropriate mental health strategies
- Creates empathetic prompts for LLM response generation

### 3. Intent Classification Module
**Purpose**: Determines the user's intention behind their message.

**Intent Categories**:
- `asking_mental_health_question`: Core mental health queries (triggers RAG)
- `greeting`: Simple hello/hi messages
- `goodbye`: Farewell statements
- `gratitude`: Thank you messages
- `out_of_scope`: Off-topic or general knowledge questions

**Key Features**:
- Hybrid classification: keyword matching + Groq LLM for ambiguous cases
- Fast lightweight fallback for common patterns
- Confidence scoring and reasoning traces

**Contribution to the System**:
- Routes non-mental-health queries to direct responses
- Triggers RAG pipeline only when needed
- Optimizes latency by avoiding unnecessary LLM calls for simple intents
- Ensures chatbot stays focused on mental health support

---

## Development Journey

### Phase 1: Exploratory Notebooks
Each module was initially developed in **Jupyter notebooks** to achieve optimal performance:
- **[Emotion_Classifier_Module.ipynb](notebooks/Emotion_Classifier_Module.ipynb)**: Fine-tuning, evaluation, and performance metrics
- **[intent_class.ipynb](notebooks/intent_class.ipynb)**: Intent pattern analysis and classification strategy
- **[language_detection.ipynb](notebooks/language_detection.ipynb)**: Language identification and multi-language testing

This approach allowed rapid experimentation and visualization of model performance before production deployment.

### Phase 2: Production Python Modules
Once optimal performance was achieved, modules were refactored into **standalone Python files** in the `modules/` directory:
- Clean, reusable class-based architecture
- Logging and error handling
- Optimized inference pipelines
- Integration-ready interfaces

### Phase 3: Unified RAG System 
All modules were orchestrated into a cohesive **RAG-based chatbot system** with:
- FastAPI web service for HTTP endpoints
- Real-time processing pipeline
- LLM-powered response generation
- Knowledge base integration

---

## RAG Architecture

The **Retrieval-Augmented Generation (RAG)** pipeline is the backbone of our chatbot's knowledge-grounded responses.

### Architecture Overview

```
User Message
    ↓
[Language Detection] → Language Code & Confidence
    ↓
[Translator] → English Translation (if needed)
    ↓
[Intent Classifier] → Intent Category
    ↓
[Emotion Classifier] → Emotional State & Tone
    ↓
[RAG Pipeline] ← Query Context
    ├─ Embed Query (Multilingual Sentence Transformer)
    ├─ Retrieve Relevant Chunks (Qdrant Vector DB)
    └─ Generate Response (Groq LLM)
    ↓
[Response Formatting] → Chat Response JSON
    ↓
User Response + Sources
```

### Key RAG Components

#### **LangChain Framework**
We leverage LangChain to automate and simplify the RAG workflow:
- **Prompt templates** for consistent, structured requests to the LLM
- **Chain orchestration** to link retrieval → context formatting → response generation
- **Built-in utilities** for token management and response parsing
- **Abstraction layers** that decouple from specific LLM providers

**Benefits**:
- Reduces boilerplate code significantly
- Enables easy swapping between LLM providers
- Structured error handling and retries
- Automatic token counting for cost optimization

#### **Multilingual Embeddings (Sentence Transformers)**
We use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:
- Converts text queries into semantic vector embeddings
- Supports 50+ languages with a single model
- Efficient (384-dimensional vectors) yet highly effective
- Fine-tuned on massive multilingual datasets for strong semantic understanding

**Advantages**:
- Enables true semantic search across languages
- Better captures meaning than keyword matching
- Faster inference compared to full-size models
- Lower memory/storage requirements

#### **Qdrant Vector Database**
High-performance vector database for storing and retrieving embeddings:
- **Cloud-hosted** for scalability and reliability
- **Fast similarity search** using HNSW (Hierarchical Navigable Small World) algorithm
- **Real-time indexing** of new knowledge base entries
- **Metadata filtering** for better ranking and context

#### **Intelligent Chunking with Q/A Pairs**
Our knowledge base is structured for optimal retrieval:
- **Semantic chunking**: Break content into meaningful units (not just by token count)
- **Q/A pair augmentation**: Each chunk includes Q&A pairs for more context aware semantic search and retrieval
- **Chunk overlap**: Slight redundancy to capture context boundaries
- **Metadata tagging**

**Example Chunk**:
```json
{
    "text":"Q: I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone? A: and that everyone has a good purpose to their life.Also, since our culture is so saturated with the belief that if someone doesn't feel good about themselves that this is somehow terrible.Bad feelings are part of living. They are the motivation to remove ourselves from situations and relationships which do us more harm than good.Bad feelings do feel terrible. Your feeling of worthlessness may be good in the sense of motivating you to find out that you are much better than your feelings today."
    "metadata":{
    "source":"mental_health_dataset"
    "chunk_id":1
    }
}
```

#### **Groq LLM Integration**
We use Groq's fast, efficient LLM for response generation:
- **High-speed inference** (tokens/sec) for real-time responses
- **Affordable API costs** with competitive pricing
- **OpenAI-compatible endpoint** for seamless integration
- **Optimized for reasoning** tasks needed in mental health conversations

---

## Unit Testing

Comprehensive unit tests ensure the reliability and correctness of all critical system components.

### Test Coverage

The test suite covers three main pillars of the system:

#### **1. Endpoints Testing** (`test_endpoints.py`)
- Tests all FastAPI endpoints including `/api/chat`, `/api/health`, and `/api/modules`
- Validates request/response formats and HTTP status codes
- Ensures proper error handling and edge case scenarios
- Verifies response time performance and payload structures

#### **2. Classifiers Testing** (`test_classifiers.py`)
- **Emotion Classifier**: Validates emotion detection accuracy across all emotion categories (sadness, joy, love, anger, fear, surprise)
- **Intent Classifier**: Tests intent classification logic including RAG-triggering intents, greetings, goodbyes, and out-of-scope detection
- **Language Detector**: Verifies multilingual language detection functionality across 40+ languages
- Confidence score validation and crisis flag detection for high-risk emotional states

#### **3. RAG Pipeline Testing** (`test_rag.py`)
- Tests the end-to-end RAG pipeline including embedding, retrieval, and response generation
- Validates retrieval quality from Qdrant vector database
- Ensures proper context window management and token counting
- Verifies source citation accuracy and relevance ranking
- Tests multilingual query handling and translation integration

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_endpoints.py
pytest tests/test_classifiers.py
pytest tests/test_rag.py

```

---

## Containerization

The application has been containerized through an iterative development process, ultimately settling on an optimized single-container approach.

### Evolution of Containerization Strategy

#### **Approach 1: Single Dockerfile (Initial)**
- Simple, monolithic containerization
- All dependencies bundled in one image
- Limitation: OpenTelemetry service wasn't running alongside the backend

#### **Approach 2: Docker Compose with Separate Services**
- Separated backend service and OpenTelemetry collector service
- Used `docker-compose.yml` for orchestration
- **Issues Encountered**:
  - Network communication overhead between containers
  - Difficulty managing service dependencies during deployment
  - Increased complexity in managing multiple container lifecycles
  - Deployment compatibility issues in certain environments

#### **Approach 3: Optimized Single Dockerfile (Final - Current)**
- **Best Solution**: Downloads OpenTelemetry collector within the same container as the backend
- Uses `start.sh` script to orchestrate both services within a single container
- **Advantages**:
  - Simpler deployment pipeline
  - Reduced network overhead
  - Single image to manage and deploy
  - Improved compatibility with containerized environments
  - Better resource efficiency

### Building and Running the Container

```bash
# Build the Docker image
docker build -t mental-health-chatbot:latest .

# Run the container
docker run -p 8000:8000 --env-file .env mental-health-chatbot:latest
```

### Container Structure
- **Base Image**: Python 3.9+
- **Included Components**:
  - FastAPI backend service
  - OpenTelemetry collector (downloaded at build time)
  - Start script (`start.sh`) to bootstrap both services
  - All Python dependencies

---

## Connecting to Axiom

The application integrates with Axiom for comprehensive observability and monitoring through OpenTelemetry.

### Axiom Integration Setup

**Axiom** is a serverless log management and analytics platform that receives telemetry data from OpenTelemetry collectors.

#### **Configuration**
- OpenTelemetry collector exports traces and logs to Axiom
- Environment configuration in `otel-collector-config.yaml`
- Axiom API endpoint and dataset configuration for data ingestion

### Metrics and Monitoring

#### **Key Metrics Collected**

1. **Request Metrics**
   - Request latency (per endpoint)
   - Request throughput (requests/sec)
   - HTTP status code distribution
   - Request payload size

2. **NLP Module Metrics**
   - Language detection accuracy and confidence scores
   - Emotion classification performance and detected emotion distribution
   - Intent classification latency and accuracy
   - Translation time (when multilingual queries are processed)

3. **RAG Pipeline Metrics**
   - Query embedding time
   - Vector database retrieval latency
   - Number of retrieved documents
   - LLM response generation time
   - End-to-end RAG pipeline latency

4. **System Health Metrics**
   - CPU and memory usage
   - API availability and uptime
   - Error rates and exception tracking
   - Crisis flag detection events

5. **Business Metrics**
   - Chat conversations count
   - Average session duration
   - User language distribution
   - Emotion distribution over time

### Accessing Axiom Dashboard
- Real-time monitoring and alerting
- Historical trend analysis
- Performance bottleneck identification
- Debugging and troubleshooting with full trace context

---

## CI/CD Pipeline

The project includes a robust Continuous Integration/Continuous Deployment pipeline for automated testing, building, and deployment.

### Pipeline Stages

#### **1. Code Quality & Testing**
- Linting and code style checks
- Unit test execution across all test files
- Test coverage validation
- Dependency vulnerability scanning

#### **2. Build & Containerization**
- Docker image building
- Image scanning for security vulnerabilities
- Tagging with version numbers and git commit SHA
- Image registry push to container repository

#### **3. Deployment**
- Automated deployment to staging environment
- Smoke tests against deployed instance
- Health check validation
- Production deployment (with manual approval)

#### **4. Post-Deployment Verification**
- Integration tests
- Performance benchmarking
- Axiom metrics validation
- Alert configuration

### CI/CD Tools
- **Version Control**: Git/GitHub
- **CI/CD Platform**: GitHub Actions
- **Container Registry**: Docker Hub / GitHub Container Registry
- **Orchestration**: Kubernetes (for advanced deployments)

### Running Locally
```bash
# Lint and format code
pylint modules/ main.py orchestrator.py
black modules/ main.py orchestrator.py

# Run tests
pytest tests/ --cov=modules

# Build image
docker build -t mental-health-chatbot:latest .
```

---

## Deployed on Hugging Face

The application is deployed and hosted on **Hugging Face Spaces**, providing easy public access and demonstration of the chatbot.

### Hugging Face Deployment

#### **Deployment Configuration**
- **Platform**: Hugging Face Spaces
- **Runtime**: Docker container
- **Port**: 8000
- **Access**: Public URL for live demonstration

#### **Benefits of Hugging Face Deployment**
- **Zero Infrastructure Management**: Automatic scaling and resource allocation
- **Easy Sharing**: Public URL for demos and stakeholder access
- **Integration**: Native support for Hugging Face models and datasets
- **Community**: Access to Hugging Face ecosystem and model library
- **Monitoring**: Built-in logging and performance monitoring

#### **Deployment Process**
1. Connect GitHub repository to Hugging Face Spaces
2. Configure environment variables (API keys, database URLs)
3. Deploy using the Dockerfile in the repository
4. Access live application via Hugging Face Space URL
5. Automatic redeploy on push to main branch

#### **Live Access**
- Application accessible via public Hugging Face Spaces URL
- Demonstration available for testing and feedback
- Easy sharing with stakeholders and users

---

## Project Structure

```
-RAG-Based-Mental-Health-Chatbot/
│
├── main.py                          # FastAPI application entry point
├── orchestrator.py                  # Main orchestration logic
├── index_qdrant.py                  # Vector DB indexing script
├── train_models.py                  # Model training pipeline
├── rag_config.json                  # RAG configuration
├── pyproject.toml                   # Project dependencies
├── README.md                        
│
├── models/                          # Pre-trained model artifacts
│   └── emotion_model/               # Fine-tuned DistilBERT
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── modules/                         # Core NLP & RAG modules
│   ├── __init__.py
│   ├── language_detector.py         # Language detection
│   ├── emotion_classifier.py        # Emotion classification
│   ├── intent_classifier.py         # Intent classification
│   ├── translator.py                # Multilingual translation
│   └── rag_pipeline.py              # RAG orchestration
│
├── notebooks/                       # Development & experimentation
│   ├── Emotion_Classifier_Module.ipynb
│   ├── intent_class.ipynb
│   └── language_detection.ipynb
│
├── templates/                       # Web UI templates
│   └── index.html                   # Main chat interface
│
├── static/                          # Static assets
│   ├── css/
│   ├── js/
│   └── images/
│
└── .env                             # Environment variables (not in repo)
    ├── groq_api_key
    ├── qdrant_url
    └── qdrant_api_key
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app with `/api/chat`, `/api/health`, `/api/modules` endpoints |
| `orchestrator.py` | Orchestrates all modules in the chat pipeline |
| `rag_pipeline.py` | RAG retrieval, ranking, and LLM response generation |
| `emotion_classifier.py` | Emotion detection with crisis flagging |
| `intent_classifier.py` | Intent classification with Groq LLM fallback |
| `language_detector.py` | Multilingual language detection |
| `translator.py` | Google Translate integration for multilingual support |

---

## Quick Start Guide

### Prerequisites

- Python 3.9+
- uv
- API Keys:
  - **Groq API** (free at [https://console.groq.com](https://console.groq.com))
  - **Qdrant Cloud** account (vector database)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/RAG-Based-Mental-Health-Chatbot.git
   cd -RAG-Based-Mental-Health-Chatbot
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # using uv:
   uv sync
   git lfs install
   git lfs pull

   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the project root:
   ```env
   groq_api_key=your_groq_api_key_here
   qdrant_url=https://your-qdrant-instance.qdrant.io:6333
   qdrant_api_key=your_qdrant_api_key_here
   ```

5. **Run the Chatbot**
   ```bash
   python -m uvicorn main:app --reload
   ```
   
   The chatbot will be available at: [http://localhost:8000](http://localhost:8000)

### Usage

#### Web Interface
1. Open [http://localhost:8000](http://localhost:8000) in your browser
2. Type your message in the chat box
3. Press Enter or click Send
4. View the response with sources and emotion detection




---

## Conclusion

This **RAG-Based Mental Health Chatbot** project represents a comprehensive integration of modern NLP and AI technologies to address real-world mental health support needs. Through this journey, we demonstrated:

**End-to-End System Design**: From exploratory notebooks to production-ready microservices

**Advanced NLP Integration**: Seamlessly combining language detection, emotion classification, and intent recognition

**Knowledge-Grounded Generation**: Leveraging RAG with LangChain, Sentence Transformers, and Qdrant for reliable, sourced responses

**Multilingual Capabilities**: Supporting 40+ languages with unified semantic understanding

**Production-Grade Architecture**: Building a scalable, maintainable system using FastAPI and cloud services

**Real-World Problem Solving**: Creating an accessible tool that can genuinely support mental health awareness and self-help

### Key Learnings
- The power of combining multiple specialized NLP models for nuanced understanding
- Effective prompt engineering for guiding LLM behavior in sensitive domains
- Vector database optimization for fast, semantic retrieval
- Orchestrating complex pipelines while maintaining code clarity and performance

### Future Enhancements
- Fine-tuned LLM specifically for mental health conversations
- Personalized response history and longitudinal mental health tracking
- Integration with professional mental health resources and crisis hotlines
- Multi-turn conversation memory for better context retention
- Mobile app deployment with offline capabilities

---


## Contributors

1. Loreen Mohamed
2. Merna Hany
3. Yasmin Yasser

