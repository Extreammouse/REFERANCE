import mujoco
import mujoco_viewer
import numpy as np
import requests
import json
import re
import os
import time

# Create a simple MuJoCo model XML
MODEL_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
    <body name="robot" pos="0 0 0.5">
      <joint type="free"/>
      <geom type="cylinder" size="0.1 0.2" rgba="0.7 0.7 0.7 1"/>
      <body name="arm" pos="0 0 0.2">
        <joint name="arm_joint" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" rgba="0.5 0.5 0.5 1"/>
        <body name="gripper" pos="0.5 0 0">
          <joint name="gripper_joint" type="slide" axis="0 0 1" range="-0.1 0"/>
          <geom type="box" size="0.1 0.05 0.05" rgba="0.3 0.3 0.3 1"/>
        </body>
      </body>
    </body>
    <!-- Objects -->
    <body name="blue_box" pos="1 0 0.1">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
    </body>
    <body name="red_cube" pos="0 1 0.05">
      <joint type="free"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="coffee_mug" pos="1 1 0.05">
      <joint type="free"/>
      <geom type="cylinder" size="0.05 0.1" rgba="0.8 0.8 0.8 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="arm_joint" name="arm_motor" gear="30"/>
    <motor joint="gripper_joint" name="gripper_motor" gear="10"/>
  </actuator>
</mujoco>
"""


class MuJoCoRobotController:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)

        # Setup viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Dictionary of objects and their body IDs
        self.objects = {
            "blue_box": self._get_body_id("blue_box"),
            "red_cube": self._get_body_id("red_cube"),
            "coffee_mug": self._get_body_id("coffee_mug")
        }

        # Points of interest
        self.points = {
            "point_a": np.array([0.5, -0.5, 0.1]),
            "point_b": np.array([0.5, 0.5, 0.1])
        }

        # Gripping state
        self.gripped_object = None

    def _get_body_id(self, body_name):
        """Get the body ID by name"""
        for i in range(self.model.nbody):
            if self.model.body_name(i) == body_name:
                return i
        return None

    def locate_object(self, object_name):
        """Find an object by name and return its position"""
        if object_name in self.objects:
            body_id = self.objects[object_name]
            if body_id is not None:
                pos = self.data.body_xpos[body_id].copy()
                return {"id": body_id, "position": pos}
        print(f"Object {object_name} not found")
        return None

    def move_to(self, x, y, z):
        """Move the robot to a specific position"""
        print(f"Moving to position: [{x}, {y}, {z}]")

        target_pos = np.array([float(x), float(y), float(z)])
        robot_body_id = self._get_body_id("robot")

        # Use a simple approach to move the robot
        steps = 100
        for i in range(steps):
            current_pos = self.data.body_xpos[robot_body_id].copy()
            direction = target_pos - current_pos
            move_step = direction / steps

            # Apply the movement
            self.data.qvel[0:3] = move_step * 10  # Linear velocity

            # Step the simulation
            mujoco.mj_step(self.model, self.data)

            # Update visualization
            self.viewer.render()

            # Move the gripped object if any
            if self.gripped_object is not None:
                obj_id = self.objects[self.gripped_object]
                # Update object position relative to gripper
                gripper_pos = self.data.body_xpos[self._get_body_id("gripper")]
                offset = np.array([0, 0, -0.15])  # Adjust based on your model
                self.data.qpos[(obj_id - 1) * 7:obj_id * 7][:3] = gripper_pos + offset

            # Short delay for visualization
            time.sleep(0.01)

        return True

    def lower_arm(self, distance_cm):
        """Lower the arm by a distance in cm"""
        print(f"Lowering arm by {distance_cm} cm")
        distance_m = float(distance_cm) / 100.0

        # Control the arm joint
        arm_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint")
        current_angle = self.data.qpos[arm_joint_id]
        target_angle = current_angle - distance_m  # Adjust based on your model

        steps = 50
        for i in range(steps):
            self.data.ctrl[0] = target_angle
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            time.sleep(0.01)

        return True

    def raise_arm(self, distance_cm):
        """Raise the arm by a distance in cm"""
        print(f"Raising arm by {distance_cm} cm")
        distance_m = float(distance_cm) / 100.0

        # Control the arm joint (opposite direction from lower_arm)
        arm_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint")
        current_angle = self.data.qpos[arm_joint_id]
        target_angle = current_angle + distance_m  # Adjust based on your model

        steps = 50
        for i in range(steps):
            self.data.ctrl[0] = target_angle
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            time.sleep(0.01)

        return True

    def close_gripper(self):
        """Close the gripper and try to grip nearby objects"""
        print("Closing gripper")

        # Control the gripper joint
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
        target_pos = -0.1  # Closed position

        steps = 30
        for i in range(steps):
            self.data.ctrl[1] = target_pos
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            time.sleep(0.01)

        # Check for nearby objects to grip
        gripper_pos = self.data.body_xpos[self._get_body_id("gripper")]
        for obj_name, obj_id in self.objects.items():
            obj_pos = self.data.body_xpos[obj_id]
            distance = np.linalg.norm(gripper_pos - obj_pos)

            if distance < 0.2:  # Gripping radius
                self.gripped_object = obj_name
                print(f"Gripped {obj_name}")
                break

        return True

    def open_gripper(self):
        """Open the gripper and release any gripped objects"""
        print("Opening gripper")

        # Control the gripper joint
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
        target_pos = 0  # Open position

        steps = 30
        for i in range(steps):
            self.data.ctrl[1] = target_pos
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            time.sleep(0.01)

        # Release any gripped object
        if self.gripped_object:
            print(f"Released {self.gripped_object}")
            self.gripped_object = None

        return True

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

                elif func_name == "pick_up" or func_name == "pick_up":
                    if len(params) == 1:
                        obj = self.locate_object(params[0])
                        if obj:
                            self.move_to(obj["position"][0], obj["position"][1], obj["position"][2] + 0.2)
                            self.lower_arm(15)
                            self.close_gripper()
                            self.raise_arm(15)
                    else:
                        print("pick_up requires 1 parameter: object_name")

                elif func_name == "put_down":
                    if len(params) == 1:
                        self.lower_arm(15)
                        self.open_gripper()
                        self.raise_arm(15)
                    else:
                        print("put_down requires 1 parameter: object_name")

                elif func_name == "lower_arm":
                    if len(params) == 1:
                        self.lower_arm(params[0])
                    else:
                        print("lower_arm requires 1 parameter: distance_cm")

                elif func_name == "raise_arm":
                    if len(params) == 1:
                        self.raise_arm(params[0])
                    else:
                        print("raise_arm requires 1 parameter: distance_cm")

                elif func_name == "close_gripper":
                    self.close_gripper()

                elif func_name == "open_gripper":
                    self.open_gripper()

                else:
                    print(f"Unknown function: {func_name}")

            except Exception as e:
                print(f"Error executing {func_name}: {e}")

        return True

    def cleanup(self):
        """Clean up resources"""
        self.viewer.close()


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
    controller = MuJoCoRobotController()
    print("Robotic environment initialized.")
    print("Type 'quit' to exit.")

    try:
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
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()