import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("robot-reasoning-api")

# Global variables
engine = None
# Semaphore to limit concurrent requests
request_semaphore = None

# Lifecycle management for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing to do here since engine is initialized separately
    yield
    # Shutdown: clean up resources
    global engine
    if engine is not None:
        logger.info("Shutting down LLM engine")
        await engine.unload_model()

app = FastAPI(title="Robot Reasoning API", lifespan=lifespan)

class RobotTaskRequest(BaseModel):
    instruction: str
    objects: List[Dict[str, List[int]]]
    # grid_size: int = 25

class RobotTaskResponse(BaseModel):
    actions: List[List[int]]  # List of [x, y, z, roll, pitch, yaw, gripper]
    raw_output: str

# Template for system prompt
SYSTEM_PROMPT="""You are a spatial reasoning assistant for a Franka Panda robot with a parallel gripper. Your task is to generate precise action sequences to accomplish object manipulation tasks.

## INPUT ENVIRONMENT:
- The workspace is a table surface represented as a 100x100 discrete grid, divided into a 25x25 grid of larger cells.
- Global positions are denoted by <|row-col|> tokens (e.g., <|3-12|>)
- When objects exist within a grid cell, their positions are further specified with <|local-row-col|> tokens (e.g., <|local-0-3|>)
- Local positions are in the range 0-3 for both row and column, representing positions in a 4x4 grid within each global cell
- Objects are represented as <|color|><|object|> tokens (e.g., <|red|><|cube|>) while <|empty|> means empty space
- Example: An object at <|5-10|><|2-3|><|red|><|cube|> is a red cube in the global cell at row 5, column 10, and within that cell, at local position row 2, column 3
- The height of each object: {object_height}

## IMPORTANT INSTRUCTIONS:
- Each output action is represented as a 7D discrete gripper action in the following format: ["<|row-col|>", "<|local-row-col|>", Z, Roll, Pitch, Yaw, Gripper] with <|row-col|> as the global position in the 25x25 grid, <|local-row-col|> as the local position within the 4x4 grid of that cell, Z is the height that gripper must reach.
- Gripper state is 0 for close and 1 for open.
- The allowed range of Z is [0, 100].
- Roll, Pitch, and Yaw are the 3D discrete orientations of the gripper in the environment, represented as discrete
Euler Angles.
- The allowed range of Roll, Pitch, and Yaw is [0, 120] and each unit represents 3 degrees.

TASK: {instruction}
{TABLE_MAP}

Think step by step about the spatial relationships and analyze the desk map to locate objects, then plan your actions step by step:
1. Identify the target object's position on the desk map.
2. Create a plan using natural language instructions that reference object tokens.
Then output ONLY the action sequence in the required format.<｜Assistant｜><think>
"""

def tokenize_desk(objects_des, grid_size=25):
    """
    Convert object positions into a tokenized desk representation with global and local positions
    
    Args:
        objects_des: List of dictionaries, each containing an object name and its [x,y,z] coordinates
                 The coordinates are in a 100x100 range
        grid_size: The size of the global grid (default: 25x25)
        
    Returns:
        A string containing the tokenized desk representation
    """
    grid = {}
    object_height = {}
    num_local_grid = 100//grid_size 
    for obj_dict in objects_des:
        for obj_name, coords in obj_dict.items():
            x, y, z = coords
            
            global_x = min(grid_size - 1, x // num_local_grid)
            global_y = min(grid_size - 1, y // num_local_grid)
            local_x = x % num_local_grid
            local_y = y % num_local_grid
            
            parts = obj_name.split("-")
            color = parts[0].strip()
            object_type = parts[1].strip()
            position = (global_x, global_y)
            grid[position] = (color, object_type, local_x, local_y)
            object_des = f"<|{color}|><|{object_type}|>"
            object_height[object_des] = z
    object_height = json.dumps(object_height)
    tokenized_desk = "<desk>\n"
    
    for row in range(grid_size):
        for col in range(grid_size):
            position = (row, col)
            if position in grid:
                color, object_type, local_row, local_col = grid[position]
                tokenized_desk += f"<|{row}-{col}|><|local-{local_row}-{local_col}|><|{color}|><|{object_type}|>"
            else:
                tokenized_desk += f"<|{row}-{col}|><|empty|>"
        tokenized_desk += "\n"
    
    tokenized_desk += "</desk>"
    return tokenized_desk, object_height

def parse_and_convert(output_text: str) -> List[List[int]]:
    """
    Parse the model output and convert to 100x100 space, returning a list of action arrays
    
    Args:
        output_text: The full output text from the model
        
    Returns:
        List of action arrays in 100x100 space format [x, y, z, roll, pitch, yaw, gripper]
    """
    step_pattern = r'Step \d+: \["<\|(\d+)-(\d+)\|>", "<\|local-(\d+)-(\d+)\|>", (\d+), (\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(step_pattern, output_text)
    
    action_sequences = []
    for match in matches:
        row, col, local_row, local_col, z, roll, pitch, yaw, gripper = map(int, match)
        
        # Convert from 25x25 to 100x100 space
        x_100 = row * 4 + local_row
        y_100 = col * 4 + local_col
        
        action = [x_100, y_100, z, roll, pitch, yaw, gripper]
        action_sequences.append(action)
    
    return action_sequences


@app.get("/health")
async def health():
    """Health check."""
    global engine
    if engine is None:
        return {"status": "initializing"}
    return {"status": "healthy"}

async def process_robot_task(request: RobotTaskRequest) -> Dict:
    """
    Process the robot task in a separate function to handle concurrency
    """
    global engine, request_semaphore
    
    try:
        # Acquire semaphore to limit concurrent requests
        async with request_semaphore:
            # Format the input using the prompt template
            print(request.objects)
            desk, object_height = tokenize_desk(request.objects)
            prompt = SYSTEM_PROMPT.format(
                object_height=object_height,
                instruction=request.instruction,
                TABLE_MAP=desk
            )
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=0.6,
                max_tokens=4096,
            )
            
            # Generate using the async engine
            request_id = random_uuid()
            results_generator = engine.generate(prompt, sampling_params, request_id)
            
            # Collect the final result (non-streaming)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
        
            if final_output is None:
                logger.error("Failed to generate output")
                return {"error": "Failed to generate output"}
                
            output_text = final_output.outputs[0].text
            discrete_actions = parse_and_convert(output_text)
            
            # Apply post-processing if we have enough actions
            if len(discrete_actions) >= 6:
                discrete_actions[5][2] = 24
                action_7 = copy.deepcopy(discrete_actions[5])
                discrete_actions.append(action_7)
                discrete_actions[5][-1] = 0
            
            return {
                "actions": discrete_actions,
                "raw_output": output_text
            }
    
    except Exception as e:
        logger.error(f"Error processing robot task: {str(e)}")
        return {"error": str(e)}

@app.post("/robot/task")
async def robot_task(request: RobotTaskRequest, background_tasks: BackgroundTasks):
    """Process robot task and return the action sequences"""
    global engine
    
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Server is still initializing"}
        )
    
    try:
        # Process the task with proper concurrency handling
        result = await process_robot_task(request)
        
        if "error" in result:
            return JSONResponse(
                status_code=500,
                content={"error": result["error"]}
            )
        
        return RobotTaskResponse(
            actions=result["actions"],
            raw_output=result["raw_output"]
        )
    
    except Exception as e:
        logger.error(f"Unhandled exception in robot_task: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal Server Error: {str(e)}"}
        )

async def initialize(model_path: str = "jan-hq/AlphaTable-1.5B", max_concurrent_requests: int = 5, **kwargs):
    """Initialize the LLM engine with the given model path"""
    global engine, request_semaphore
    
    try:
        logger.info(f"Initializing LLM engine with model {model_path}")
        
        # Create a semaphore to limit concurrent requests
        request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype="bfloat16",
            **kwargs
        )
        
        # Initialize the engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("LLM engine initialization complete")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize engine: {str(e)}")
        raise

def start_server(host="0.0.0.0", port=8000, model_path="jan-hq/AlphaTable-1.5B", 
                max_concurrent_requests=5, **kwargs):
    """Start the server with the given host and port"""
    import uvicorn
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize the engine
    try:
        loop.run_until_complete(initialize(
            model_path=model_path, 
            max_concurrent_requests=max_concurrent_requests, 
            **kwargs
        ))
        
        # Start the server
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot Reasoning API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=3348, help="Port to bind the server to")
    parser.add_argument("--model", type=str, default="jan-hq/AlphaTable-1.5B-v0.2", help="Model path or name")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--max-concurrent-requests", type=int, default=10, 
                      help="Maximum number of concurrent requests to process")
    
    args = parser.parse_args()
    
    try:
        start_server(
            host=args.host,
            port=args.port,
            model_path=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_concurrent_requests=args.max_concurrent_requests
        )
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")