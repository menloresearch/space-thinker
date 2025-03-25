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