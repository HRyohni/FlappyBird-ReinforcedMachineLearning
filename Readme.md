# Flappy Bird Reinforced learning



This project is focused on training an agent to play the popular game Flappy Bird using reinforcement learning. The goal is to develop an intelligent agent that can navigate through the game environment and achieve high scores.

## Abstract

The project starts by setting up the necessary dependencies and packages. It installs required libraries like matplotlib, numpy, tqdm, and gymnasium. These libraries provide functionalities for visualization, numerical computations, progress tracking, and the game environment. Next, the code defines a class called `FlappyBirdAgent`, which represents the learning agent. This agent uses Q-learning, a type of reinforcement learning algorithm, to learn the best actions to take in different game states. The agent maintains a dictionary of state-action values called `q_values` and utilizes a set of hyperparameters like learning rate, epsilon (exploration rate), and discount factor. The `FlappyBirdAgent` class has methods for selecting actions (`get_action`), updating the Q-values based on observed rewards and state transitions (`update`), and decaying the exploration rate (`decay_epsilon`) over time. After defining the agent, the code sets the hyperparameters and creates an instance of the agent. It also initializes variables for tracking scores during training. The reinforcement learning process starts by creating the Flappy Bird game environment. Then, a loop runs for a specified number of episodes. In each episode, the agent interacts with the environment by selecting actions, observing rewards and state transitions, and updating its knowledge. The loop continues until the episode is terminated or truncated. During training, the code keeps track of the scores achieved by the agent in each episode. The `scoreTimeLine` list stores the scores, allowing for later analysis and visualization. Finally, the code visualizes the training progress by plotting various graphs using the matplotlib library. It displays the moving average of episode rewards, episode lengths, training error, and the score timeline. In summary, this project demonstrates the application of reinforcement learning techniques to train an agent to play Flappy Bird. It showcases the agent's learning process, tracks scores, and visualizes the training progress. Overall, the code sets up the Flappy Bird environment, defines a reinforcement learning agent, runs the training process, and visualizes the training progress.

## Rendering

enviroment can be shown in real time enabling **env.render()** and **sleep** method.

![](https://cdn.discordapp.com/attachments/1083688555534098473/1118322534606192660/ezgif.com-video-to-gif.gif)

## Statistics

Statistics can be shown at the end of learning. Statistics show **Episode reward**, **Episode lengths**, **Traning Error** and **Score Timeline** 

![](https://cdn.discordapp.com/attachments/1083688555534098473/1118973198340472932/Figure_1.png)



# Explanation of code

```python
# installing assets
#get_ipython().system('pip install matplotlib')
#get_ipython().system('pip install numpy')
#get_ipython().system('pip install tqdm')
#get_ipython().system('pip install gymnasium==0.27.0')
#get_ipython().system('pip install flappy_bird_gymnasium')
#get_ipython().run_line_magic('matplotlib', 'inline')
```

> These commands are usually executed in a Jupyter Notebook.



```python
# imports
import matplotlib as plt
import numpy as np
from tqdm import tqdm
import flappy_bird_gymnasium
import time
import gymnasium as gym 
from collections import defaultdict
```

> These lines import the necessary libraries and modules for the code. It imports `matplotlib` as `plt`, `numpy` as `np`, `tqdm`, `flappy_bird_gymnasium`, `time`, `gymnasium` as `gym`, and `defaultdict` from the `collections` module.



```python
env = gym.make("FlappyBird-v0")
```

> This line creates the Flappy Bird environment using the `gym.make()` function and assigns it to the variable `env`. The environment is created with the ID `"FlappyBird-v0"`.



```python
class FlappyBirdAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

```

> This is the definition of the `FlappyBirdAgent` class. It initializes an agent with the given parameters: `learning_rate`, `initial_epsilon`, `epsilon_decay`, `final_epsilon`, and `discount_factor`. It also initializes the agent's `q_values` dictionary, `lr` (learning rate), `discount_factor`, `epsilon` (exploration rate), `epsilon_decay`, `final_epsilon`, and `training_error` list.



```python
def get_action(self, obs) -> float:
    """
    Returns the best action with probability (1 - epsilon)
    otherwise a random action with probability epsilon to ensure exploration.
    """
    obs_tuple = tuple(obs)  # Convert obs to tuple

    # with probability epsilon return a random action to explore the environment
    if np.random.random() < self.epsilon:
        return env.action_space.sample()

    # with probability (1 - epsilon) act greedily (exploit)
    else:
        return int(np.argmax(self.q_values[obs_tuple]))

```

> This method `get_action` returns the best action based on the agent's current state observation (`obs`). With a probability of `epsilon`, it returns a random action to explore the environment. Otherwise, it selects the action with the highest Q-value for the given state (`obs`) from the `q_values` dictionary.



```python
def update(
    self,
    obs,
    action: int,
    reward: float,
    terminated: bool,
    next_obs,
):
    """Updates the Q-value of an action."""
    obs_tuple = tuple(obs)  # Convert obs to tuple
    next_obs_tuple = tuple(next_obs)  # Convert next_obs to tuple

    future_q_value = (not terminated) * np.max(self.q_values[next_obs_tuple])
    temporal_difference = (
        reward + self.discount_factor * future_q_value - self.q_values[obs_tuple][action]
    )

    self.q_values[obs_tuple][action] = (
        self.q_values[obs_tuple][action] + self.lr * temporal_difference
    )
    self.training_error.append(temporal_difference)

```

> This method `update` is used to update the Q-value of an action based on the observed reward and the next state. It calculates the temporal difference error and updates the Q-value using the Q-learning update rule.



```python
def decay_epsilon(self):
    self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
```

> This method `decay_epsilon` is used to decay the exploration rate (`epsilon`) over time by subtracting `epsilon_decay` from `epsilon` until it reaches the `final_epsilon`.



```python
# Settings

# hyperparameters
learning_rate = 0.0001
n_episodes = 1000000
start_epsilon = 0.1
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.001

agent = FlappyBirdAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

```

> These lines define the hyperparameters and create an instance of the `FlappyBirdAgent` class with the specified hyperparameters.



```python
env = gym.make("FlappyBird-v0")
agent = FlappyBirdAgent(learning_rate=0.1, initial_epsilon=1.0, epsilon_decay=0.001, final_epsilon=0.01)
scoreTimeLine = []
scoreTemp = -1
from IPython.display import clear_output
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)
        
        done = terminated or truncated
        if terminated:
            scoreTimeLine.append(info["score"])
            scoreTemp = -1
        obs = next_obs

    agent.decay_epsilon()

```

> These lines run the reinforcement learning process. It loops through `n_episodes` episodes, where each episode consists of multiple steps. In each step, it selects an action based on the agent's current state, updates the agent's knowledge using the `update` method, and transitions to the next state. It also keeps track of the scores during each episode.



```python
import matplotlib.pyplot as plt
import numpy as np

print("score List: ", len(scoreTimeLine))
#print(scoreTimeLine)

rolling_length = 500

fig, axs = plt.subplots(ncols=4, figsize=(16, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

axs[3].set_title("Score Timeline")
axs[3].plot(range(len(scoreTimeLine)), scoreTimeLine)

plt.tight_layout()
plt.show()

```

> These lines visualize the training progress by plotting various graphs. It plots the moving average of episode rewards, episode lengths, training error, and the score timeline.