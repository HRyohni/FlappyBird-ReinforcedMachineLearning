import time
import flappy_bird_gymnasium
from gymnasium.wrappers import GrayScaleObservation
import gymnasium as gym
from stable_baselines3 import PPO
    #consts
gameName = "FlappyBird-rgb-v0"
env = gym.make(gameName)

def logs():
    log_path = os.path.join('Traning','Logs')
    env = gym.make(enviroment_name)
    env = DummyVecEnv([lambda: env])
    model = PPO ('MlpPolicy',env,verbose=1,tensorboard_log=log_path)



obs, _ = env.reset()

#env = GrayScaleObservation(env, keep_dim=True)

while True:
    # Next action:
  
    # (feed the observation to your agent here)
    action = env.action_space.sample()
   #print(env.step(1))
    # Processing:
    obs, reward, terminated, _, info = env.step(action)

   
    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS
    
    # Checking if the player is still alive
    if terminated:
        env.reset()
        

env.close()


if __name__ == "__main__":
    main()