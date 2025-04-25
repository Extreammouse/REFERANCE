import json
import os


def convert_robothor_to_training_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    training_examples = []

    # Extract episode information
    episode_id = data["episode_id"]
    scene = data["scene"]
    goal = data["goal"]

    # Create a natural language description of the goal
    goal_description = f"Find the {goal['target_object']} and move it to position x={goal['target_position']['x']}, y={goal['target_position']['y']}, z={goal['target_position']['z']}"

    # Create examples from steps
    for step in data["steps"]:
        # Create natural language instruction
        instruction = ""
        if step["action"] == "MoveAhead":
            instruction = f"Move forward to position x={step['target_position']['x']}, y={step['target_position']['y']}, z={step['target_position']['z']}"
        elif step["action"] == "Look":
            instruction = f"Turn to look at position x={step['target_position']['x']}, y={step['target_position']['y']}, z={step['target_position']['z']}"
        else:
            instruction = f"Perform {step['action']} to position x={step['target_position']['x']}, y={step['target_position']['y']}, z={step['target_position']['z']}"

        # Add object interactions if any
        for interaction in step["object_interactions"]:
            instruction += f" and {interaction['interaction_type']} the {interaction['object_type']}"

        # Create corresponding robotic command
        response = f"move_to({step['target_position']['x']}, {step['target_position']['y']}, {step['target_position']['z']})"

        # Add interaction commands
        for interaction in step["object_interactions"]:
            if interaction["interaction_type"] == "picked_up":
                response += f"; identify_object('{interaction['object_type']}'); pick_up('{interaction['object_type']}')"
            elif interaction["interaction_type"] == "put_down":
                response += f"; put_down('{interaction['object_type']}')"

        # Add to training examples
        training_examples.append({
            "instruction": instruction,
            "response": response
        })

    # Add a high-level goal example
    training_examples.append({
        "instruction": goal_description,
        "response": f"locate_object('{goal['target_object']}'); move_to(object_position); pick_up('{goal['target_object']}'); move_to({goal['target_position']['x']}, {goal['target_position']['y']}, {goal['target_position']['z']}); put_down('{goal['target_object']}')"
    })

    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(training_examples, f, indent=2)

    return len(training_examples)


# Example usage
input_file = "robothor_episode.json"  # Your RoboTHOR data file
output_file = "training_data.json"  # Output file for training

# Save your RoboTHOR data to this file first
with open(input_file, 'w') as f:
    f.write('''
{
  "episode_id": "1",
  "scene": "FloorPlan1",
  "steps": [
    {
      "step_id": 1,
      "action": "MoveAhead",
      "action_success": true,
      "agent_position": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0
      },
      "target_position": {
        "x": 1.0,
        "y": 0.0,
        "z": 0.0
      },
      "object_interactions": [
        {
          "object_id": "object_1",
          "object_type": "chair",
          "interaction_type": "picked_up",
          "interaction_success": true
        }
      ]
    },
    {
      "step_id": 2,
      "action": "Look",
      "action_success": true,
      "agent_position": {
        "x": 1.0,
        "y": 0.0,
        "z": 0.0
      },
      "target_position": {
        "x": 1.5,
        "y": 0.0,
        "z": 0.0
      },
      "object_interactions": []
    }
  ],
  "goal": {
    "goal_id": "goal_1",
    "target_object": "coffee_mug",
    "target_position": {
      "x": 2.0,
      "y": 0.0,
      "z": 1.0
    }
  }
}
''')

num_examples = convert_robothor_to_training_data(input_file, output_file)
print(f"Created {num_examples} training examples in {output_file}")