import gym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "airgym:airsim-drone-sample-v0", 
        ip_address="127.0.0.1", 
        image_shape=(50,50,3)
    )
)])

env = VecTransposeImage(env)

model = PPO(
    'CnnPolicy', 
    env, 
    verbose=1, 
    seed=42,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=4,
    best_model_save_path=".",
    log_path=".",
    eval_freq=500,
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

#log_name = "ppo_run_" + str(time.time())

model.learn(
    total_timesteps=100000,
    tb_log_name="first_run",
    **kwargs
)

model.save("ppo_navigation_policy")
