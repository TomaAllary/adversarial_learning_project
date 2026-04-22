# The custom PPO implementation has been replaced by Stable Baselines 3.
# Training is now done via pettingzoo_env/train_ppo.py using SB3's PPO.
#
# To train:
#     python -m pettingzoo_env.train_ppo
#
# To load a saved model:
#     from stable_baselines3 import PPO
#     model = PPO.load("runs/ppo_<timestamp>/best_model.zip")
#     obs, _ = env.reset()
#     action, _ = model.predict(obs, deterministic=True)
