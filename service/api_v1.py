import re
import json
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import copy

app = FastAPI(title="Robot Reasoning API")
engine = None

class RobotTaskRequest(BaseModel):
    instruction: str
    objects: List[Dict[str, List[int]]]
    # grid_size: int = 25

class RobotTaskResponse(BaseModel):
    actions: List[List[int]]  # List of [x, y, z, roll, pitch, yaw, gripper]
    raw_output: str

# Template for system prompt
SYSTEM_PROMPT = """You are a spatial reasoning assistant for a Franka Panda robot with a parallel gripper. Your task is to generate precise action sequences to accomplish object manipulation tasks.

## INPUT ENVIRONMENT:
- The workspace is a table surface represented as a 25x25 discrete grid for visualization
- Each position is denoted by <|row-col|> tokens (e.g., <|3-12|>)
- Objects are represented as <|color|><|object|> tokens (e.g., <|red|><|cube|>) while <|empty|> means empty space.
- The height of each object: {object_height}

## IMPORTANT INSTRUCTIONS:
- Each output action is represented as a 6D discrete gripper action in the following format: ["<row-col>", Z, Roll, Pitch, Yaw,
Gripper] with <row-col> is the discrete positions of gripper on the table and Z is the height that gripper must reach.
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

def tokenize_desk(objects_des: List[Dict[str, List[int]]], grid_size: int = 25) -> tuple:
    """
    Convert object positions into a tokenized desk representation
    
    Args:
        objects_des: List of dictionaries, each containing an object name and its [x,y,z] coordinates
                 The coordinates are in a 100x100 range and will be quantized to a 25x25 grid
        grid_size: The size of the quantized grid (default: 25x25)
        
    Returns:
        A tuple containing (tokenized desk representation, object heights dictionary)
    """
    grid = {}
    object_height = {}
    
    for obj_dict in objects_des:
        for obj_name, coords in obj_dict.items():
            x, y, z = coords
            
            quantized_x = min(grid_size - 1, x * grid_size // 100)
            quantized_y = min(grid_size - 1, y * grid_size // 100)
            
            parts = obj_name.split("-")
            color = parts[0].strip()
            object_type = parts[1].strip()
            position = (quantized_y, quantized_x)
            grid[position] = (color, object_type)
            object_des = f"<|{color}|><|{object_type}|>"
            object_height[object_des] = z
    
    tokenized_desk = "<desk>\n"
    
    for row in range(grid_size):
        for col in range(grid_size):
            position = (row, col)
            if position in grid:
                color, object_type = grid[position]
                tokenized_desk += f"<|{row}-{col}|><|{color}|><|{object_type}|>"
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
    step_pattern = r'Step \d+: \["<\|(\d+)-(\d+)\|>", (\d+), (\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(step_pattern, output_text)
    
    action_sequences = []
    for match in matches:
        row, col, z, roll, pitch, yaw, gripper = map(int, match)
        
        # Convert from 25x25 to 100x100 space
        x_100 = col * 4 + 2
        y_100 = row * 4 + 2
        
        action = [x_100, y_100, z, roll, pitch, yaw, gripper]
        action_sequences.append(action)
    
    return action_sequences

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}

@app.post("/robot/task", response_model=RobotTaskResponse)
async def robot_task(request: RobotTaskRequest):
    """Process robot task and return the action sequences"""
    global engine
    
    try:
        # Format the input using the prompt template
        desk, object_height = tokenize_desk(request.objects)
        prompt = SYSTEM_PROMPT.format(
            object_height=json.dumps(object_height),
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
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to generate output"}
            )
            
        output_text = final_output.outputs[0].text
        discrete_actions = parse_and_convert(output_text)
        discrete_actions[5][2] += 1
        action_7 = copy.deepcopy(discrete_actions[5])
        discrete_actions.append(action_7)
        discrete_actions[5][-1] = 0
        # print(discrete_actions[5][-1])
        # print(action_7)
        
        return RobotTaskResponse(
            actions=discrete_actions,
            raw_output=output_text
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

async def initialize(model_path: str = "jan-hq/AlphaTable-1.5B", **kwargs):
    """Initialize the LLM engine with the given model path"""
    global engine
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        dtype="bfloat16",
        **kwargs
    )
    
    # Initialize the engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return app

def start_server(host="0.0.0.0", port=8000, model_path="jan-hq/AlphaTable-1.5B", **kwargs):
    """Start the server with the given host and port"""
    import uvicorn
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize the engine
    loop.run_until_complete(initialize(model_path, **kwargs))
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot Reasoning API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=3348, help="Port to bind the server to")
    parser.add_argument("--model", type=str, default="jan-hq/AlphaTable-1.5B", help="Model path or name")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum model length")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )