import time
import gc
import re
import torch
from typing import Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

# Initialize the FastAPI application
app = FastAPI(title="Hugging Face Local API", description="API for running ML models locally")

# Cache to store already-loaded models
models_cache = {}

# Prompt template for the SLR (Systematic Literature Review) selection task.
# The placeholders {title}, {abstract}, and {criteria} are filled at request time.
SLR_SELECTION_PROMPT_TEMPLATE = "On a scale from 1 (very low probability) to 5 (very high probability), how would you rate the relevance of the scientific publication for inclusion into a systematic literature review based on the relevant criteria and based on title and abstract? Title: “{title}” Abstract: “{abstract}” Relevant Criteria: “{criteria}” Answer:"

def get_model(model_name: str, hf_token: Optional[str] = None, enable_gpu: bool = False):
    cache_key = f"{model_name}_{'gpu' if enable_gpu else 'cpu'}"
    if cache_key not in models_cache:
        print(f"Loading model {model_name} (GPU: {enable_gpu}), this may take a moment...")
        try:
            device = 0 if enable_gpu else -1
            
            kwargs = {}
            if enable_gpu:
                # Use float16 to halve the VRAM requirement on GPU
                kwargs["dtype"] = torch.float16
                
            models_cache[cache_key] = pipeline(
                "text-generation", 
                model=model_name, 
                token=hf_token, 
                device=device,
                **kwargs
            )
            print(f"Model {model_name} loaded successfully on {'GPU' if enable_gpu else 'CPU'}!")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model '{model_name}': {str(e)}")
    return models_cache[cache_key]

@app.post("/clear-cache")
def clear_cache():
    """
    Clears the loaded models from memory and empties the CUDA cache.
    Use this if you encounter CUDA Out of Memory errors.
    """
    global models_cache
    models_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"message": "Model cache cleared and GPU memory freed."}

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

class SlrSelectionResponse(BaseModel):
    prompt: str
    likert_score: float = Field(..., alias="likert-score")

@app.post("/slr-selection-generate-global", response_model=SlrSelectionResponse)
def slr_selection_generate_global_structured(request: SlrSelectionRequest):
    """
    Endpoint that builds an SLR selection prompt and returns a structured output
    containing ONLY the prompt and the extracted likert score as a float.
    """
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
        
        # Extract the score from the generated output
        generated_text = result[0]['generated_text']
        # Look only at the newly generated text if it includes the prompt
        new_text = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
        
        # Simple regex to find a number between 1 and 5 (including floats)
        match = re.search(r'\b([1-5](?:\.\d+)?)\b', new_text)
        score = float(match.group(1)) if match else 0.0

        return SlrSelectionResponse(
            prompt=prompt,
            **{"likert-score": score}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SlrSelectionPerCriteriaRequest(BaseModel):
    title: str
    abstract: str
    criteria: list[str]
    model_name: str = ""
    hf_token: Optional[str] = None
    max_length: int = 50
    enable_gpu: bool = True
    debug: bool = True

class SlrSelectionPerCriteriaResponse(BaseModel):
    prompt: list[str]
    likert_score: list[float] = Field(..., alias="likert-score")

@app.post("/slr-selection-generate-per-criteria", response_model=SlrSelectionPerCriteriaResponse)
def slr_selection_generate_per_criteria(request: SlrSelectionPerCriteriaRequest):
    """
    Endpoint that builds SLR selection prompts iteratively for a list of criteria,
    runs inference on each, and returns a list of likert scores corresponding to each criterion.
    """
    try:
        if not request.title.strip():
            raise HTTPException(status_code=400, detail="Title must not be empty")
        if not request.abstract.strip():
            raise HTTPException(status_code=400, detail="Abstract must not be empty")
        if not request.criteria:
            raise HTTPException(status_code=400, detail="Criteria list must not be empty")

        generator = get_model(request.model_name, request.hf_token, request.enable_gpu)

        prompts = []
        scores = []

        for criterion in request.criteria:
            if not criterion.strip():
                prompts.append("")
                scores.append(0.0)
                continue

            # Build the final prompt by injecting the inputs into the template
            prompt = SLR_SELECTION_PROMPT_TEMPLATE.format(
                title=request.title,
                abstract=request.abstract,
                criteria=criterion,
            )
            prompts.append(prompt)

            # Run the model with additional parameters
            result = generator(prompt, max_length=request.max_length, num_return_sequences=1)
            
            # Extract the score from the generated output
            generated_text = result[0]['generated_text']
            new_text = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
            
            # Simple regex to find a number between 1 and 5 (including floats)
            match = re.search(r'\b([1-5](?:\.\d+)?)\b', new_text)
            score = float(match.group(1)) if match else 0.0
            
            scores.append(score)

        return SlrSelectionPerCriteriaResponse(
            prompt=prompts,
            **{"likert-score": scores}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PostprocessingRequest(BaseModel):
    likert_score: Union[float, list[float]] = Field(..., alias="likert-score")
    threshold: float
    logic: str = "AND"

class PostprocessingResponse(BaseModel):
    label: str

@app.post("/postprocessing", response_model=PostprocessingResponse)
def postprocessing(request: PostprocessingRequest):
    """
    Endpoint to assign 'relevant' or 'irrelevant' labels based on likert-score(s)
    and a given threshold. Works for both a single score and a list of scores.
    If a list is provided, it aggregates the boolean results using the specified logic (AND/OR).
    """
    try:
        if isinstance(request.likert_score, list):
            # Check threshold for each score
            bool_results = [score >= request.threshold for score in request.likert_score]
            
            # Aggregate using the specified logic
            if request.logic.upper() == "OR":
                final_result = any(bool_results)
            else:
                final_result = all(bool_results)  # default to AND
                
            label = "relevant" if final_result else "irrelevant"
            return PostprocessingResponse(label=label)
        else:
            label = "relevant" if request.likert_score >= request.threshold else "irrelevant"
            return PostprocessingResponse(label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


