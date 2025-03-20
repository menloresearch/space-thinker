import requests
import json
from typing import List, Dict

class RobotTaskClient:
    """Client for interacting with the Robot Reasoning API."""

    def __init__(self, base_url: str):
        """
        Initialize the client.

        Args:
            base_url: The base URL of the API server (e.g., "http://localhost:8000").
        """
        self.base_url = base_url

    def health_check(self) -> Dict:
        """Check the health of the API server."""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()

    def send_task(self, instruction: str, objects: List[Dict[str, List[int]]]) -> Dict:
        """
        Send a robot task to the API and get the response.

        Args:
            instruction: The task instruction.
            objects: A list of dictionaries, each representing an object with its name and [x, y, z] coordinates.

        Returns:
            A dictionary containing the 'actions' (list of action arrays) and 'raw_output' (the raw model output).
            Raises an exception if the request fails.
        """
        url = f"{self.base_url}/robot/task"
        headers = {"Content-Type": "application/json"}
        data = {
            "instruction": instruction,
            "objects": objects,
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
def example_usage():
    """Demonstrates how to use the RobotTaskClient."""

    # --- Configuration ---
    client = RobotTaskClient(base_url="http://localhost:8000")

    # --- Health Check ---
    try:
        health = client.health_check()
        print(f"Health Check: {health}")
    except requests.exceptions.RequestException as e:
        print(f"Health Check Failed: {e}")
        return 

    # --- Example Task: Single object ---
    instruction = "Stack the black cube on top of the red cube"
    objects = [
        {"red-cube": [51, 43, 17]},
        {"black-cube": [44, 58, 17]},
        {"purple-cube": [74, 59, 17]},
        {"green-cube": [65, 82, 17]},
    ]

    try:
        response = client.send_task(instruction, objects)
        print(f"\nTask Response:\nActions: {response['actions']}\nRaw Output:\n{response['raw_output']}")
    except requests.exceptions.RequestException as e:
        print(f"Task Failed: {e}")

if __name__ == "__main__":
    example_usage()