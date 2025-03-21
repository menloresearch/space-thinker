import random
import json
import math
import argparse
from datasets import Dataset
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
- Each output action is represented as a 7D discrete gripper action in the following format: ["<|row-col|>", "<|local-row-col|>", Z, Roll, Pitch, Yaw, Gripper] with <|row-col|> as the global position in the 25x25 grid, <|local-row-col|> as the local position within the 4x4 grid of that cell, Z is the height from Gripper Tip to Table surface.
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
Then output ONLY the action sequence in the required format.
"""
objects = ["moon", "star", "cube", "cylinder", "triangular prism"]
colors = ["red", "maroon", "lime", "green", "blue", "navy", "yellow", "cyan", "magenta", "silver", "gray", "olive", "purple", "teal", "azure", "violet", "rose", "black", "white"]

Thinking_Format_Stack = """LOCATE OBJECTS:
Target Object: {source_object} found at {source_pos} with height {source_height}
Target Placement Location: {target_object} at {target_pos} with height {target_height}
PLAN ACTIONS:
Step 1: Move above source_object at source_pos with > source height.
Step 2: Position the gripper at the base of the source_object on the table surface.
Step 3: Close gripper at the same position to pick up source_object.
Step 4: Lift source_object at source_pos to height > target height.
Step 5: Move to target_object location at target_pos: {target_pos} at the same height as stage 4.
Step 6: Move on top of target_object location at target_pos with height target_height + 1 to avoid collision.
Step 7: Open gripper to finish the task.
"""

Thinking_Format_Place = """LOCATE OBJECTS:
Target Object: {source_object} found at {source_pos} with height {source_height}
Target Placement Location: {target_object} at {target_pos} with height {target_height}
PLAN ACTIONS:
Step 1: Move above source_object at source_pos with > source height.
Step 2: Position the gripper at the base of the source_object on the table surface.
Step 3: Close gripper at the same position to pick up source_object.
Step 4: Lift source_object at source_pos to height > target height.
Step 5: Move to target_object location at target_pos: {target_pos} at the same height as stage 4.
Step 6: Move on top of target_object location at target_pos with height target_height cause the container is empty.
Step 7: Open gripper to drop the {source_object} into {target_object}.
"""

Thinking_Format_Move = """LOCATE OBJECTS:
Target Object: {source_object} found at {source_pos} with height {source_height}
Target Placement Location: {target_con_pos} with height {target_height}. Map {target_con_pos} (100x100) to a 25x25 grid, then a 4x4 subgrid. Result: {target_pos}.
PLAN ACTIONS:
Step 1: Move above source_object at source_pos with > source height.
Step 2: Position the gripper at the base of the source_object on the table surface.
Step 3: Close gripper at the same position to pick up source_object.
Step 4: Lift source_object at source_pos to height > target height.
Step 5: Move to target_object location at target_pos: {target_pos} at the same height as stage 4.
Step 6: Move on top of target location at target_pos: {target_pos} with height target_height.
Step 7: Open gripper to finish the task.
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

def convert_solution(actions, to_tokenized=True):
    """
    Convert a solution from 100x100 format to 25x25 format with tokenized positioning
    
    Args:
        actions: List of 7D actions in format [x_100, y_100, z, roll, pitch, yaw, gripper]
        to_tokenized: If True, convert to <row-col> format, otherwise use (row,col)
        
    Returns:
        List of converted actions
    """
    converted_actions = []
    
    for action in actions:
        x_100, y_100, z, roll, pitch, yaw, gripper = action
        
        # Convert from 100x100 to 25x25
        x_25 = x_100 // 4
        y_25 = y_100 // 4
        local_x_25 = x_100 % 4
        local_y_25 = y_100 % 4
        
        if to_tokenized:
            # Format as <|row-col|>
            position = f"<|{x_25}-{y_25}|>"
            local_pos = f"<|local-{local_x_25}-{local_y_25}|>"
            converted_action = [position, local_pos, z, roll, pitch, yaw, gripper]
        else:
            # Format as tuple (row,col)
            converted_action = [(x_25, y_25), (local_x_25, local_y_25), z, roll, pitch, yaw, gripper]
            
        converted_actions.append(converted_action)
    
    return converted_actions

def discretize_object(objects_pos: list):
    x_100, y_100, z = objects_pos
    x_25 = x_100 // 4
    y_25 = y_100 // 4
    local_x_25 = x_100 % 4
    local_y_25 = y_100 % 4
    position = f"<|{x_25}-{y_25}|><|local-{local_x_25}-{local_y_25}|>"
    converted_object = [position, z]
    return converted_object

def generate_task_unique(task_type):
    """
    Generate synthetic robotic data samples with unique objects (except containers).
    
    Args:
        task_type: Type of task to generate (placing, move, stack)
        
    Returns:
        Dictionary of generated data sample
    """
    global objects, colors, SYSTEM_PROMPT, Thinking_Format_Stack, Thinking_Format_Place
    
    num_objects = random.randint(4, 6)
    scene_objects = []
    used_descriptions = set()
    positions = []
    source_obj = []
    target_obj = []
    used_object_types = set()
    
    # Setup target object (container for placing task)
    if task_type == "placing":
        target_object_type = "container"
    else:
        target_object_type = random.choice(objects)
        used_object_types.add(target_object_type)
        
    target_color = random.choice(colors)
    target_desc = f"{target_color}-{target_object_type}"
    target_x = random.randint(0, 98)
    target_y = random.randint(0, 98)
    target_z = random.randint(1, 30)
    target_position = [target_x, target_y, target_z]
    target_discrete_pos = discretize_object(target_position)
    scene_objects.append({target_desc: target_position})
    target_obj.append({f"<|{target_color}|><|{target_object_type}|>": target_discrete_pos})
    used_descriptions.add(target_desc)
    positions.append((target_x, target_y))
    
    available_objects = [obj for obj in objects if obj not in used_object_types or obj == "container"]
    if not available_objects:
        available_objects = ["container"]  # Nếu hết loại đối tượng thì dùng container
        
    source_object_type = random.choice(available_objects)
    if source_object_type != "container":
        used_object_types.add(source_object_type)
        
    source_color = random.choice(colors)
    source_desc = f"{source_color}-{source_object_type}"
    
    while source_desc in used_descriptions:
        source_color = random.choice(colors)
        source_desc = f"{source_color}-{source_object_type}"
    
    source_x, source_y = generate_position_with_min_distance(positions, 4)
    source_z = random.randint(1, 30)
    source_position = [source_x, source_y, source_z]
    source_discrete_pos = discretize_object(source_position)
    scene_objects.append({source_desc: source_position})
    source_obj.append({f"<|{source_color}|><|{source_object_type}|>": source_discrete_pos})
    used_descriptions.add(source_desc)
    positions.append((source_x, source_y))
    
    # Add 1-2 additional containers with different colors for placing task
    if task_type == "placing":
        num_extra_containers = random.randint(1, 2)
        for _ in range(num_extra_containers):
            extra_container_color = random.choice(colors)
            extra_container_desc = f"{extra_container_color}-container"
            
            # Ensure we don't duplicate container colors
            while extra_container_desc in used_descriptions:
                extra_container_color = random.choice(colors)
                extra_container_desc = f"{extra_container_color}-container"
            
            extra_x, extra_y = generate_position_with_min_distance(positions, 4)
            extra_z = random.randint(1, 30)
            
            scene_objects.append({extra_container_desc: [extra_x, extra_y, extra_z]})
            used_descriptions.add(extra_container_desc)
            positions.append((extra_x, extra_y))
    
    # Calculate how many more objects to add
    remaining_objects = num_objects - 2  # source and target already added
    if task_type == "placing":
        remaining_objects -= num_extra_containers  # account for the extra containers
    
    # Add remaining additional objects (ensuring uniqueness except for containers)
    for _ in range(remaining_objects):
        available_objects = [obj for obj in objects if obj not in used_object_types or obj == "container"]
        if not available_objects:
            available_objects = ["container"] 
            
        obj = random.choice(available_objects)
        if obj != "container":
            used_object_types.add(obj)
            
        color = random.choice(colors)
        desc = f"{color}-{obj}"
        
        while desc in used_descriptions:
            color = random.choice(colors)
            desc = f"{color}-{obj}"
        
        x, y = generate_position_with_min_distance(positions, 4)
        z = random.randint(1, 30)
        
        scene_objects.append({desc: [x, y, z]})
        used_descriptions.add(desc)
        positions.append((x, y))
    
    random.shuffle(scene_objects)
    
    # Create instruction based on task type
    if task_type == "placing":
        instruction = f"Pick up the {source_object_type} and place it into the {target_color} {target_object_type}"  
                       
    elif task_type == "move":
        instruction = f"Move the {source_object_type} to {json.dumps(target_position)}"
    else: 
        instruction_list = [f"Stack the {source_object_type} on top of the {target_object_type}",
                       f"Stack the {target_object_type} and the {source_object_type} in sequence.",
                       ]
        instruction = instruction_list[random.randint(0,1)]
    
    roll, pitch, yaw = 0, 60, 90
    
    # Calculate end position based on task type
    if task_type == "placing" or task_type == "move":
        end_z = target_z
    else: 
        end_z = target_z + 1  # Position slightly above the target for stacking
    
    solutions = [
        [source_x, source_y, random.randint(source_z+10, max(source_z+10, 15)), roll, pitch, yaw, 1],  # Approach with gripper open
        [source_x, source_y, 0, roll, pitch, yaw, 1],  # Move to object with gripper open
        [source_x, source_y, 0, roll, pitch, yaw, 0],  # Close gripper to grasp object
        [source_x, source_y, random.randint(source_z+10, max(source_z+10, 15)), roll, pitch, yaw, 0],  # Lift object with gripper closed
        [target_x, target_y, random.randint(source_z+10, max(source_z+10, 15)), roll, pitch, yaw, 0],  # Move above target with gripper closed
        [target_x, target_y, end_z, roll, pitch, yaw, 0],
        [target_x, target_y, end_z, roll, pitch, yaw, 1]  # Open gripper to release object
    ]
    converted_solution = convert_solution(solutions)
    if task_type=="placing":
        think_answer = Thinking_Format_Place.format(source_object=f"<|{source_color}|><|{source_object_type}|>", source_pos=source_discrete_pos[0], source_height=source_discrete_pos[1], target_object=f"<|{target_color}|><|{target_object_type}|>", target_pos=target_discrete_pos[0], target_height=target_discrete_pos[1])
    elif task_type=="move":
        think_answer = Thinking_Format_Move.format(source_object=f"<|{source_color}|><|{source_object_type}|>", source_pos=source_discrete_pos[0], source_height=source_discrete_pos[1], target_con_pos=target_position[:2], target_pos=target_discrete_pos[0], target_height=target_discrete_pos[1])
    else:
        think_answer = Thinking_Format_Stack.format(source_object=f"<|{source_color}|><|{source_object_type}|>", source_pos=source_discrete_pos[0], source_height=source_discrete_pos[1], target_object=f"<|{target_color}|><|{target_object_type}|>", target_pos=target_discrete_pos[0], target_height=target_discrete_pos[1])
    answer=""
    for i, solution in enumerate(converted_solution):
        solution_str = json.dumps(solution)
        if i == len(converted_solution) - 1:
            answer +=f"Step {i+1}: {solution_str}"
            break
        answer +=f"Step {i+1}: {solution_str}\n"
    final_answer=f"<think>\n{think_answer}\n</think>\n\n{answer}"
    desk, object_height = tokenize_desk(scene_objects)
    text = SYSTEM_PROMPT.format(object_height=object_height,instruction=instruction,TABLE_MAP=desk)
    user_part = {"content": text.strip(), "role": "user"}
    assistant_part = {"content": final_answer.strip(), "role": "assistant"}
    data_sample = {
        "Source_Obj": json.dumps(source_obj),
        "Target_Obj": json.dumps(target_obj),
        "Thinking": think_answer,
        "Object": json.dumps(scene_objects),
        "instruction": instruction,
        "solution": solutions,
        "Conversation": [user_part, assistant_part]
    }
    
    return data_sample

def generate_task(task_type):
    """
    Generate synthetic robotic data samples.
    
    Args:
        num_samples: Number of data samples to generate
        
    Returns:
        List of generated data samples
    """
    global objects, colors, SYSTEM_PROMPT, Thinking_Format_Stack, Thinking_Format_Place
    
    num_objects = random.randint(5, 7)
    scene_objects = []
    used_descriptions = set()
    positions = []
    source_obj = []
    target_obj = []
    
    # Setup target object (container for placing task)
    if task_type == "placing":
        target_object_type = "container"
    else:
        target_object_type = random.choice(objects)
    target_color = random.choice(colors)
    target_desc = f"{target_color}-{target_object_type}"
    target_x = random.randint(0, 98)
    target_y = random.randint(0, 98)
    target_z = random.randint(1, 30)
    target_position = [target_x, target_y, target_z]
    target_discrete_pos = discretize_object(target_position)
    scene_objects.append({target_desc: target_position})
    target_obj.append({f"<|{target_color}|><|{target_object_type}|>": target_discrete_pos})
    used_descriptions.add(target_desc)
    positions.append((target_x, target_y))
    
    # Setup source object
    source_object_type = random.choice(objects)
    source_color = random.choice(colors)
    source_desc = f"{source_color}-{source_object_type}"
    
    while source_desc in used_descriptions:
        source_color = random.choice(colors)
        source_object_type = random.choice(objects)
        source_desc = f"{source_color}-{source_object_type}"
    
    source_x, source_y = generate_position_with_min_distance(positions, 4)
    source_z = random.randint(1, 30)
    source_position = [source_x, source_y, source_z]
    source_discrete_pos = discretize_object(source_position)
    scene_objects.append({source_desc: source_position})
    source_obj.append({f"<|{source_color}|><|{source_object_type}|>": source_discrete_pos})
    used_descriptions.add(source_desc)
    positions.append((source_x, source_y))
    
    # Add 1-2 additional containers with different colors for placing task
    if task_type == "placing":
        num_extra_containers = random.randint(1, 2)
        for _ in range(num_extra_containers):
            extra_container_color = random.choice(colors)
            extra_container_desc = f"{extra_container_color}-container"
            
            # Ensure we don't duplicate container colors
            while extra_container_desc in used_descriptions:
                extra_container_color = random.choice(colors)
                extra_container_desc = f"{extra_container_color}-container"
            
            extra_x, extra_y = generate_position_with_min_distance(positions, 4)
            extra_z = random.randint(1, 30)
            
            scene_objects.append({extra_container_desc: [extra_x, extra_y, extra_z]})
            used_descriptions.add(extra_container_desc)
            positions.append((extra_x, extra_y))
    
    # Calculate how many more objects to add
    remaining_objects = num_objects - 2  # source and target already added
    if task_type == "placing":
        remaining_objects -= num_extra_containers  # account for the extra containers
    
    # Add remaining additional objects
    for _ in range(remaining_objects):
        obj = random.choice(objects)
        color = random.choice(colors)
        desc = f"{color}-{obj}"
        
        while desc in used_descriptions:
            color = random.choice(colors)
            obj = random.choice(objects)
            desc = f"{color}-{obj}"
        
        x, y = generate_position_with_min_distance(positions, 4)
        z = random.randint(1, 30)
        
        scene_objects.append({desc: [x, y, z]})
        used_descriptions.add(desc)
        positions.append((x, y))
    
    random.shuffle(scene_objects)
    
    # Create instruction based on task type
    if task_type == "placing":
        instruction = f"Pick up the {source_color} {source_object_type} and place it into the {target_color} {target_object_type}"  
                       
    elif task_type == "move":
        instruction = f"Move the {source_color} {source_object_type} to {json.dumps(target_position)}"
    else: 
        instruction_list = [f"Stack the {source_color} {source_object_type} on top of the {target_color} {target_object_type}",
                       f"Stack the {target_color} {target_object_type} and the {source_color} {source_object_type} in sequence.",
                       ]
        instruction = instruction_list[random.randint(0,1)]
    
    roll, pitch, yaw = 0, 60, 90
    
    # Calculate end position based on task type
    if task_type == "placing" or task_type == "move":
        end_z = target_z
    else: 
        end_z = target_z + 1  # Position slightly above the target for stacking
    
    solutions = [
        [source_x, source_y, random.randint(source_z+10, max(source_z+10, 15)), roll, pitch, yaw, 1],  # Approach with gripper open
        [source_x, source_y, 0, roll, pitch, yaw, 1],  # Move to object with gripper open
        [source_x, source_y, 0, roll, pitch, yaw, 0],  # Close gripper to grasp object
        [source_x, source_y, random.randint(source_z+10, max(source_z+10, 15)), roll, pitch, yaw, 0],  # Lift object with gripper closed
        [target_x, target_y, random.randint(source_z+10, max(source_z+10, 15)), roll, pitch, yaw, 0],  # Move above target with gripper closed
        [target_x, target_y, end_z, roll, pitch, yaw, 0],
        [target_x, target_y, end_z, roll, pitch, yaw, 1]  # Open gripper to release object
    ]
    converted_solution = convert_solution(solutions)
    if task_type=="placing":
        think_answer = Thinking_Format_Place.format(source_object=f"<|{source_color}|><|{source_object_type}|>", source_pos=source_discrete_pos[0], source_height=source_discrete_pos[1], target_object=f"<|{target_color}|><|{target_object_type}|>", target_pos=target_discrete_pos[0], target_height=target_discrete_pos[1])
    elif task_type=="move":
        think_answer = Thinking_Format_Move.format(source_object=f"<|{source_color}|><|{source_object_type}|>", source_pos=source_discrete_pos[0], source_height=source_discrete_pos[1], target_con_pos=target_position[:2], target_pos=target_discrete_pos[0], target_height=target_discrete_pos[1])
    else:
        think_answer = Thinking_Format_Stack.format(source_object=f"<|{source_color}|><|{source_object_type}|>", source_pos=source_discrete_pos[0], source_height=source_discrete_pos[1], target_object=f"<|{target_color}|><|{target_object_type}|>", target_pos=target_discrete_pos[0], target_height=target_discrete_pos[1])
    answer=""
    for i, solution in enumerate(converted_solution):
        solution_str = json.dumps(solution)
        if i == len(converted_solution) - 1:
            answer +=f"Step {i+1}: {solution_str}"
            break
        answer +=f"Step {i+1}: {solution_str}\n"
    final_answer=f"<think>\n{think_answer}\n</think>\n\n{answer}"
    desk, object_height = tokenize_desk(scene_objects)
    text = SYSTEM_PROMPT.format(object_height=object_height,instruction=instruction,TABLE_MAP=desk)
    user_part = {"content": text.strip(), "role": "user"}
    assistant_part = {"content": final_answer.strip(), "role": "assistant"}
    data_sample = {
        "Source_Obj": json.dumps(source_obj),
        "Target_Obj": json.dumps(target_obj),
        "Thinking": think_answer,
        "Object": json.dumps(scene_objects),
        "instruction": instruction,
        "solution": solutions,
        "Conversation": [user_part, assistant_part]
    }
    
    return data_sample

def generate_position_with_min_distance(existing_positions, min_distance):
    while True:
        x = random.randint(0, 98)
        y = random.randint(0, 98)
        
        if all((abs(x - pos[0]) >= min_distance or abs(y - pos[1]) >= min_distance) for pos in existing_positions):
            return x, y

def generate_robotic_data(num_placing_samples=5, num_stacking_samples=5, num_move_samples=5, number_unique_placing=70000, number_unique_stacking=30000):
    data_samples = []
    
    for _ in range(num_placing_samples):
        data_samples.append(generate_task("placing"))
    
    for _ in range(num_stacking_samples):
        data_samples.append(generate_task("stacking"))
        
    for _ in range(num_move_samples):
        data_samples.append(generate_task("move"))
        
    for _ in range(number_unique_placing):
        data_samples.append(generate_task_unique("placing"))
    for _ in range(number_unique_stacking):
        data_samples.append(generate_task_unique("stacking"))
    random.shuffle(data_samples)  # Shuffle to mix up the task types
    def transform_list_to_dict(list_of_dicts):
        result = {}
        keys = list_of_dicts[0].keys()
        for key in keys:
            result[key] = [item[key] for item in list_of_dicts]
        
        return result
    dict_format = transform_list_to_dict(data_samples)
    dataset = Dataset.from_dict(dict_format)
    dataset.push_to_hub("jan-hq/Pick-Place-Table-Reasoning-local-pos-v0.2", split="train")
    return data_samples

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic robotic task data.')
    parser.add_argument('--placing', type=int, default=100000, help='Number of placing task samples')
    parser.add_argument('--stacking', type=int, default=120000, help='Number of stacking task samples')
    parser.add_argument('--moving', type=int, default=40000, help='Number of stacking task samples')
    parser.add_argument('--output', type=str, default='synthetic_robotic_data.json', help='Output file name')
    
    args = parser.parse_args()
    
    data_samples = generate_robotic_data(args.placing, args.stacking, args.moving)
    
    print(f"Generated {len(data_samples)} synthetic robotic data samples")
    print(f" - Placing tasks: {args.placing}")
    print(f" - Stacking tasks: {args.stacking}")
    
    print("\nSample task:")
    print(json.dumps(data_samples[0], indent=2))
    
    with open(args.output, 'w') as f:
        json.dump(data_samples, f, indent=2)
    
    print(f"\nAll samples saved to '{args.output}'")
    

if __name__ == "__main__":
    main()
    objects = [
    {"purple-cube": [27, 29, 18]},
    {"blue-container": [76, 65, 17]},
    {"purple-triangular prism": [51, 55, 18]},
    {"orange-star": [57, 65, 17]}
    ]

    tokenized_output, object_height = tokenize_desk(objects)
    print(object_height)
