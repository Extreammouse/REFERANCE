import requests
import json
import time

def train_ollama_model(model_name, training_data_file):
    """
    Train an Ollama model using a series of prompts and completions
    """
    # Load training data
    with open(training_data_file, 'r') as f:
        training_data = json.load(f)

    total_examples = len(training_data)

    for i, example in enumerate(training_data):
        print(f"Training example {i + 1}/{total_examples}")

        # Extract the instruction and response
        instruction = example["instruction"]
        response = example["response"]

        # Format for Ollama
        prompt = f"Instruction: {instruction}\nResponse: "

        # Using Ollama's learning capability
        try:
            # First, generate a completion
            generate_response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'system': 'Convert natural language instructions into precise robotic control sequences.',
                    'stream': False
                }
            )

            # If the response doesn't match our expected output, we teach the model
            if generate_response.json()['response'].strip() != response.strip():
                create_response = requests.post(
                    'http://localhost:11434/api/create',
                    json={
                        'name': model_name,
                        'prompt': prompt + response,
                        'system': 'Convert natural language instructions into precise robotic control sequences.',
                        'stream': False
                    }
                )

                if create_response.status_code != 200:
                    print(f"Error training example {i + 1}: {create_response.text}")

            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing example {i + 1}: {e}")

    print(f"Training completed with {total_examples} examples")
    return True

# Usage:
train_ollama_model('robot-arm', 'robot_commands.json')