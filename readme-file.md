# Ollama-Powered Robotic Arm Controller

This project uses an Ollama LLM to translate natural language commands into robotic arm control sequences for object manipulation tasks in PyBullet.

## Requirements

- Python 3.8+
- Ollama installed
- External SSD (recommended for model storage)
- PyBullet
- CUDA-compatible GPU (recommended but not required)

## Installation

1. **Install Ollama**

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Install Python Dependencies**

   ```bash
   pip install pybullet numpy requests
   ```

3. **Configure External SSD**

   ```bash
   # Mount your external SSD
   sudo mkdir -p /mnt/ollama_ssd
   sudo mount /dev/sdX /mnt/ollama_ssd
   sudo chown -R $USER:$USER /mnt/ollama_ssd

   # Configure Ollama to use the SSD
   mkdir -p /mnt/ollama_ssd/ollama
   echo 'export OLLAMA_MODELS=/mnt/ollama_ssd/ollama' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Download Base Model**

   ```bash
   ollama pull llama2:7b
   ```

## Training the Model

1. **Prepare Training Data**

   Create a file named `robot_commands.json` with training examples:

   ```json
   [
     {
       "instruction": "Pick up the blue box",
       "response": "locate_object('blue box'); move_arm_to(blue_box.position); lower_arm(5); close_gripper(); raise_arm(10)"
     },
     ...
   ]
   ```

2. **Create the Modelfile**

   ```
   FROM llama2:7b
   PARAMETER temperature 0.7
   PARAMETER top_p 0.9
   PARAMETER stop "</s>"
   PARAMETER stop "Instruction:"
   PARAMETER stop "Response:"

   SYSTEM """
   You are a robotic arm control assistant. Convert natural language instructions into precise robotic arm control sequences.
   """

   TEMPLATE """
   <s>[INST] {{ .System }} [/INST]</s>

   [INST] Instruction: {{ .Instruction }} [/INST]
   Response: {{ .Response }}
   """
   ```

3. **Create and Train the Model**

   ```bash
   ollama create robot-arm -f Modelfile
   ```

   Use the API for training:

   ```bash
   python -c "
   import requests, json
   with open('robot_commands.json', 'r') as f:
       data = json.load(f)
   for example in data:
       requests.post('http://localhost:11434/api/train', json={
           'name': 'robot-arm',
           'data': [example]
       })
   "
   ```

## Running the Robot Controller

1. **Start Ollama Server**

   ```bash
   ollama serve
   ```

2. **Run the PyBullet Integration Script**

   ```bash
   python robot_controller.py
   ```

3. **Issue Commands**

   When prompted, enter natural language commands like:
   - "Pick up the blue box"
   - "Move the red cube to point A"
   - "Stack the red cube on top of the blue box"

## Fine-tuning Parameters

Important parameters to consider when training:

- **Temperature**: Controls randomness (0.2-0.7 recommended)
- **Learning rate**: Use low values (1e-5 to 5e-5) to prevent catastrophic forgetting
- **Context length**: At least 2048 tokens for complex instructions
- **Top-p**: 0.9 is a good starting point for balanced outputs

## Troubleshooting

- **Memory Issues**: Reduce model size or increase swap space
- **Slow Inference**: Enable GPU acceleration or optimize parameters
- **Poor Command Translation**: Add more examples to the training dataset

## Extending the Project

- Add computer vision for object detection
- Implement reinforcement learning for improved performance
- Create a web interface for remote control
- Support multiple robotic platforms

## License

MIT
