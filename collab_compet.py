import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from time import sleep

from ddpg_agent import Agent

from unityagents import UnityEnvironment

def ddpg(n_episodes, num_agents=1, slow_every=100, slow_by=None):
    all_scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes +1):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for i in range(2):
          agents[i].reset()
        t = 0
        while True:
            t += 1
            actions1 = agents[0].act(states[0]) # select an action (for each agent)
            actions2 = agents[1].act(states[1])
            actions1 = np.clip(actions1, -1, 1)                  # all actions between -1 and 1
            actions2 = np.clip(actions2, -1, 1)
            #print(actions1, actions2)
            env_info = env.step(np.concatenate((actions1, actions2)))[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agents[0].step(states[0], actions1, rewards[0], next_states[0], dones)
            agents[1].step(states[1], actions2, rewards[1], next_states[1], dones)
            scores += np.sum(rewards)                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if slow_by is not None and i_episode % slow_every == 0:
              sleep(slow_by)
            if np.any(dones):                                  # exit loop if episode finished
                break
        print(scores, t)
        scores_window.append(scores)       # save most recent score
        all_scores.append(scores)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=1.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_checkpoint.pth')
            return all_scores
    return all_scores

def load_agent():
  actor_state_dict = torch.load('actor_checkpoint.pth')
  critic_state_dict = torch.load('critic_checkpoint.pth')
  agent.actor_local.load_state_dict(actor_state_dict)
  agent.actor_target.load_state_dict(actor_state_dict)
  agent.critic_local.load_state_dict(critic_state_dict)
  agent.critic_target.load_state_dict(critic_state_dict)

def watch_one_episode(slow_by=None):
  env_info = env.reset(train_mode=True)[brain_name] # reset the environment
  state = env_info.vector_observations[0]            # get the current state
  score = 0
  while True:
    action = agent.act(state)                 # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    state = next_state                             # roll over the state to next time step
    score += reward
    print('\rScore: {}'.format(score), end="")
    if slow_by is not None:
      sleep(slow_by)
    if done:                                       # exit loop if episode finished
      break
  print()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Solve the Reacher environment using DDPG!')
  parser.add_argument('--visualize', dest='visualize', action='store_true', default=False,
                    help='Watch agent learn environment')
  parser.add_argument('--no_train', dest='train', action='store_false', default=True,
                    help='add this if you don\'t want to train the agent')
  parser.add_argument('--load', dest='load', action='store_true', default=False,
                    help='Load a saved model')
  parser.add_argument('--watch_one_episode', dest='watch_one_episode', action='store_true', default=False,
                    help='watch one episode of the agent in action')
  parser.add_argument('--n_episodes', dest='n_episodes', type=int, default=10000,
                    help='max number of episodes to train the agent')
  parser.add_argument('--actor_hidden_layers', dest='actor_hidden_layers', nargs='+', type=int, default=[400,300],
                    help='list of hidden layer sizes for the actor')
  parser.add_argument('--critic_hidden_layers', dest='critic_hidden_layers', nargs='+', type=int, default=[400,300],
                    help='list of hidden layer sizes for the critic')
  parser.add_argument('--ou_theta', dest='ou_theta', type=float, default=0.15,
                    help='Theta parameter in Ornstein-Uhlenbeck process')
  parser.add_argument('--ou_theta_decay', dest='ou_theta_decay', type=float, default=1.0,
                    help='Factor to decay OU theta by after each episode')  
  parser.add_argument('--ou_sigma', dest='ou_sigma', type=float, default=0.2,
                    help='Sigma parameter in Ornstein-Uhlenbeck process')
  parser.add_argument('--ou_sigma_decay', dest='ou_sigma_decay', type=float, default=1.0,
                    help='Factor to decay OU sigma by after each episode')    
  parser.add_argument('--energy_penalty', dest='energy_penalty', type=float, default=0.0,
                    help='Weight for L1 penalty on actions to discourage wasted energy')            
  parser.add_argument('--slow_every', dest='slow_every', type=int, default=100,
                    help='number of episodes before watching the agent slowly')
  parser.add_argument('--slow_by', dest='slow_by', type=float, default=None,
                    help='time to sleep between steps to better visualize environment')

  args = parser.parse_args()

  num_agents = 1

  file_name = './Tennis_Linux/Tennis'
  env = UnityEnvironment(file_name=file_name)

  # get the default brain
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]
  # reset the environment
  env_info = env.reset(train_mode=True)[brain_name]
  agents = []
  for _ in range(2):
    agents.append(Agent(
      state_size=len(env_info.vector_observations[0])
      , action_size=brain.vector_action_space_size
      , actor_hidden_layers=args.actor_hidden_layers
      , critic_hidden_layers=args.critic_hidden_layers
      , ou_theta=args.ou_theta
      , ou_sigma=args.ou_sigma
      , ou_theta_decay=args.ou_theta_decay
      , ou_sigma_decay=args.ou_sigma_decay
      , energy_penalty=args.energy_penalty
      , random_seed=1
    ))

  if args.load:
    load_agent()
  if args.watch_one_episode:
    watch_one_episode(args.slow_by)
  if args.train:
    scores = ddpg(
      n_episodes=args.n_episodes
      , slow_every=args.slow_every
      , slow_by=args.slow_by
    )
    print(scores)
    outfile = open('scores.txt', 'w')
    outfile.write(','.join(list(map(str, scores))))
    outfile.close()

    env.close()
