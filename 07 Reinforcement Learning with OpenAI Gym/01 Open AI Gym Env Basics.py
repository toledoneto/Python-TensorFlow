import gym


# criando o ambiente
env = gym.make('CartPole-v0')

# resetando o env
env.reset()

print("Observação inicial")
observation = env.reset()
print(observation)

for _ in range(2):
    # # renderizando o env
    # env.render()

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    print("Performed One Random Action")
    print('\n')
    print('observation')
    print(observation)
    print('\n')

    print('reward')
    print(reward)
    print('\n')

    print('done')
    print(done)
    print('\n')

    print('info')
    print(info)
    print('\n')
