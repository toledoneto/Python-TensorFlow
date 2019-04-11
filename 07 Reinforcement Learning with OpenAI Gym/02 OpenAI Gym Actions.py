import gym
env = gym.make('CartPole-v0')


# print(env.action_space.)  # Discrete(2)
# print(env.observation_space)  # Box(4,)
observation = env.reset()

for t in range(1000):

    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    # Move o carro pro lado que o bastão estiver caindo

    # angulo é medido a partir da vertical
    if pole_ang > 0:
        # para dir
        action = 1
    else:
        # para esq
        action = 0

    # Realizando a ação
    observation, reward, done, info = env.step(action)
