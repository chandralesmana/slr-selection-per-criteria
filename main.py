import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI application
app = FastAPI(title="Hugging Face Local API", description="API for running ML models locally")

# Cache to store already-loaded models
models_cache = {}

# Prompt template for the SLR (Systematic Literature Review) selection task.
# The placeholders {title}, {abstract}, and {criteria} are filled at request time.
SLR_SELECTION_PROMPT_TEMPLATE = """
On a scale from 1 (very low probability) to 5 (very high probability), how would you rate the relevance of the scientific publication for inclusion into a systematic literature review based on the relevant criteria and based on title and abstract? 
Title: “{title}” 
Abstract: “{abstract}” 
Relevant Criteria: 
“{criteria}”
Answer:
"""

def get_model(model_name: str, hf_token: Optional[str] = None, enable_gpu: bool = False):
    cache_key = f"{model_name}_{'gpu' if enable_gpu else 'cpu'}"
    if cache_key not in models_cache:
        print(f"Loading model {model_name} (GPU: {enable_gpu}), this may take a moment...")
        try:
            device = 0 if enable_gpu else -1
            models_cache[cache_key] = pipeline("text-generation", model=model_name, token=hf_token, device=device)
            print(f"Model {model_name} loaded successfully on {'GPU' if enable_gpu else 'CPU'}!")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model '{model_name}': {str(e)}")
    return models_cache[cache_key]

# Request schema using Pydantic
class TextRequest(BaseModel):
    text: str
    model_name: str = ""
    hf_token: Optional[str] = None
    max_length: int = 50
    enable_gpu: bool = True
    debug: bool = True

@app.get("/")
def read_root():
    return {"message": "API is running. Visit /docs for the UI documentation."}

@app.post("/text-generate-raw")
def generate_text(request: TextRequest):
    """
    Endpoint that accepts an initial prompt and generates a text continuation.
    """
    start_time = time.time() if request.debug else None
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")
        
        # Retrieve the model from cache or load a new one
        generator = get_model(request.model_name, request.hf_token, request.enable_gpu)
        
        # Run the model with additional parameters
        result = generator(request.text, max_length=request.max_length, num_return_sequences=1)
        
        response = {
            "prompt": request.text,
            "model_name": request.model_name,
            "generated_text": result
        }
        
        if request.debug:
            response["debug_info"] = {
                "execution_time_seconds": time.time() - start_time,
                "device_used": "gpu" if request.enable_gpu else "cpu",
            }
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SlrSelectionRequest(BaseModel):
    title: str
    abstract: str
    criteria: str
    model_name: str = ""
    hf_token: Optional[str] = None
    max_length: int = 50
    enable_gpu: bool = True
    debug: bool = True


@app.post("/slr-selection-generate-global-raw")
def slr_selection_generate(request: SlrSelectionRequest):
    """
    Endpoint that builds an SLR selection prompt from the given title, abstract,
    and criteria, then generates the model's inclusion/exclusion decision.
    """
    start_time = time.time() if request.debug else None
    try:
        if not request.title.strip():
            raise HTTPException(status_code=400, detail="Title must not be empty")
        if not request.abstract.strip():
            raise HTTPException(status_code=400, detail="Abstract must not be empty")
        if not request.criteria.strip():
            raise HTTPException(status_code=400, detail="Criteria must not be empty")

        # Build the final prompt by injecting the inputs into the template
        prompt = SLR_SELECTION_PROMPT_TEMPLATE.format(
            title=request.title,
            abstract=request.abstract,
            criteria=request.criteria,
        )

        # Retrieve the model from cache or load a new one
        generator = get_model(request.model_name, request.hf_token, request.enable_gpu)

        # Run the model with additional parameters
        result = generator(prompt, max_length=request.max_length, num_return_sequences=1)

        response = {
            "title": request.title,
            "abstract": request.abstract,
            "criteria": request.criteria,
            "prompt": prompt,
            "model_name": request.model_name,
            "generated_text": result,
        }

        if request.debug:
            response["debug_info"] = {
                "execution_time_seconds": time.time() - start_time,
                "device_used": "gpu" if request.enable_gpu else "cpu",
            }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
