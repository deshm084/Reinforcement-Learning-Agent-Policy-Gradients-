# 🎮 Reinforcement Learning Agent (Policy Gradients)

An implementation of a Reinforcement Learning agent that learns to play games from scratch using the **REINFORCE (Policy Gradient)** algorithm. 

Currently configured to master the **CartPole-v1** environment in OpenAI's Gymnasium.

## 🎯 The Goal
To build an AI that doesn't just memorize data, but learns through trial and error.
* **Input:** The state of the game (Position, Velocity, Angle).
* **Output:** The probability of taking a specific action (Left vs. Right).
* **Feedback:** +1 Reward for every frame the pole stays upright.

## 🧠 How It Works
Unlike Supervised Learning, there are no "labels." The agent explores random actions and learns from consequences using **Policy Gradients**:

1.  **The Policy Network:** A PyTorch Neural Network approximates the best action to take.
2.  **Discounted Rewards:** The agent calculates the long-term value of an action using a discount factor ($\gamma = 0.99$).
    $$R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} ...$$
3.  **Backpropagation:** We update the network weights to increase the probability of actions that led to high rewards and decrease those that led to failure.

## 🛠 Tech Stack
* **Python 3.8+**
* **PyTorch** (Neural Network & Automatic Differentiation)
* **Gymnasium** (Game Environment)
* **NumPy** (Math operations)
* **Matplotlib** (Visualization)

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install gymnasium torch numpy matplotlib
    ```
2.  Run the training script:
    ```bash
    python rl_agent.py
    ```
3.  Watch the training loop. The agent usually solves the environment (Score 495+) within **500-800 episodes**.

## 📊 Results
* **Episodes 0-100:** Random flailing (Avg Score: 20).
* **Episodes 200-400:** Learning balance strategies (Avg Score: 100+).
* **Episodes 500+:** Mastery (Avg Score: 495+).

## 💡 Key Learnings
* **Reward Engineering:** Learned that RL agents are extremely sensitive to how rewards are structured.
* **Exploration vs. Exploitation:** The probabilistic nature of the Policy Network handles exploration naturally without needing complex Epsilon-Greedy strategies.
