import pybullet as p
import pybullet_data
import time
import numpy as np
import requests
import json
import re


class RobotThorController:
    def __init__(self):
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load environment (floor)
        self.plane_id = p.loadURDF("plane.urdf")

        # Load a simple robot (can be replaced with more complex models)
        self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

        # Add objects to the environment
        self.load_objects()

        # Store objects with their IDs
        self.objects = {}

    def load_objects(self):
        # Add a coffee mug
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.05,
            length=0.1,
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.05,
            height=0.1
        )
        coffee_mug_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[1.5, 0.5, 0.05]
        )
        self.objects["coffee_mug"] = {"id": coffee_mug_id, "position": [1.5, 0.5, 0.05]}

        # Add a chair
        chair_id = p.loadURDF("table/table.urdf", [1.0, 0.0, 0.0], globalScaling=0.5)
        self.objects["chair"] = {"id": chair_id, "position": [1.0, 0.0, 0.0]}

    def move_to(self, x, y, z):
        """Move the robot to a specific position"""
        print(f"Moving to position: [{x}, {y}, {z}]")

        # Current position
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)

        # Calculate direction vector
        direction = [x - pos[0], y - pos[1], 0]  # Keep z unchanged for now
        distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2)

        if distance > 0.01:  # Only move if not already at position
            steps = int(distance * 100)  # More steps for smoother movement
            for i in range(steps):
                ratio = (i + 1) / steps
                new_pos = [
                    pos[0] + direction[0] * ratio,
                    pos[1] + direction[1] * ratio,
                    z  # Set desired z
                ]
                p.resetBasePositionAndOrientation(self.robot_id, new_pos, orn)
                p.stepSimulation()
                time.sleep(0.01)

        return True

    def locate_object(self, object_name):
        """Find an object by name"""
        if object_name in self.objects:
            # Get current position from PyBullet
            if "id" in self.objects[object_name]:
                pos, _ = p.getBasePositionAndOrientation(self.objects[object_name]["id"])
                self.objects[object_name]["position"] = pos
            return self.objects[object_name]
        else:
            print(f"Object {object_name} not found")
            return None

    def pick_up(self, object_name):
        """Pick up an object"""
        obj = self.locate_object(object_name)
        if obj:
            print(f"Picking up {object_name}")
            # Create a constraint to attach the object to the robot
            constraint_id = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=-1,
                childBodyUniqueId=obj["id"],
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.2],
                childFramePosition=[0, 0, 0]
            )
            self.objects[object_name]["constraint"] = constraint_id
            return True
        return False

    def put_down(self, object_name):
        """Put down a held object"""
        obj = self.locate_object(object_name)
        if obj and "constraint" in obj:
            print(f"Putting down {object_name}")
            p.removeConstraint(obj["constraint"])
            del self.objects[object_name]["constraint"]
            return True
        return False

    def execute_command(self, command):
        """Execute a command string from the LLM"""
        print(f"Executing: {command}")

        # Split into individual commands
        commands = re.split(r';\s*', command)

        for cmd in commands:
            if not cmd:
                continue

            # Extract function name and parameters
            match = re.match(r'(\w+)\((.*)\)', cmd)
            if not match:
                print(f"Could not parse command: {cmd}")
                continue

            func_name, params_str = match.groups()

            # Parse parameters
            params = []
            if params_str:
                # Handle quoted strings
                parts = []
                current = ""
                in_quotes = False
                for char in params_str:
                    if char == "'" or char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        parts.append(current.strip())
                        current = ""
                    else:
                        current += char
                if current:
                    parts.append(current.strip())

                # Process parameters
                for part in parts:
                    # Remove quotes
                    if (part.startswith("'") and part.endswith("'")) or \
                            (part.startswith('"') and part.endswith('"')):
                        part = part[1:-1]

                    # Try to convert to numeric if possible
                    try:
                        if '.' in part:
                            part = float(part)
                        else:
                            part = int(part)
                    except ValueError:
                        pass

                    params.append(part)

            # Execute the function
            try:
                if func_name == "move_to":
                    if len(params) == 3:
                        self.move_to(params[0], params[1], params[2])
                    else:
                        print("move_to requires 3 parameters: x, y, z")

                elif func_name == "locate_object":
                    if len(params) == 1:
                        self.locate_object(params[0])
                    else:
                        print("locate_object requires 1 parameter: object_name")

                elif func_name == "pick_up":
                    if len(params) == 1:
                        self.pick_up(params[0])
                    else:
                        print("pick_up requires 1 parameter: object_name")

                elif func_name == "put_down":
                    if len(params) == 1:
                        self.put_down(params[0])
                    else:
                        print("put_down requires 1 parameter: object_name")

                else:
                    print(f"Unknown function: {func_name}")

            except Exception as e:
                print(f"Error executing {func_name}: {e}")

        return True


def query_ollama(instruction):
    """Query the robot-arm model"""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'robot-arm',
            'prompt': f'Instruction: {instruction}',
            'system': 'Convert natural language instructions into precise robotic control sequences.'
        }
    )
    return response.json()['response']


def main():
    controller = RobotThorController()
    print("Robotic environment initialized.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nEnter instruction: ")
        if user_input.lower() == 'quit':
            break

        # Get command sequence from LLM
        try:
            llm_response = query_ollama(user_input)
            print(f"LLM response: {llm_response}")

            # Execute the command
            controller.execute_command(llm_response)
        except Exception as e:
            print(f"Error: {e}")

    # Disconnect from PyBullet
    p.disconnect()


if __name__ == "__main__":
    main()