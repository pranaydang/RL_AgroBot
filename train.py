import gymnasium as gym
from torch_nn import Model
import numpy as np
import torch

def sample_action(probs):

    idx = probs[0].multinomial(num_samples=1)
    action = idx.item()

    return action, probs[0, idx]

def build_policy_model():
    
    return Model()
    
def loss_fn(rewards, prob_selected, gamma, device):

    loss = torch.tensor(0.0, device=device)
    prob_selected = torch.cat(prob_selected).squeeze()
    # mean = torch.mean(rewards).unsqueeze()
    returns = []

    for i in range(len(rewards)):
        r_k = 0
        for k in range(i,len(rewards)):
            r_k += (gamma**(k-i))*rewards[k]
        returns.append(r_k)
    
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    
    returns -= returns.mean()

    loss = -torch.sum(returns * torch.log(prob_selected + 1e-8))

    return loss

def run_episode(env, policy_model, device):

    prob_selected = []
    rewards = []

    obs, info = env.reset()

    done = False

    while not done:

        # Convert observation from env to tensor
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float()
        obs_tensor /= 255.0

        # forward pass
        probs = policy_model(obs_tensor)

        # sample
        action, prob = sample_action(probs)

        # environment step
        obs, reward, done, truncated, info = env.step(action)

        # store
        prob_selected.append(prob)
        rewards.append(reward)

        if done or truncated:
            break
    
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    # prob_selected = torch.tensor(prob_selected, dtype=torch.float32, device=device)

    return prob_selected, rewards

def train(env, policy_model, optimizer, n_episodes):

    device = next(policy_model.parameters()).device

    for episode in range(n_episodes):

        print(episode)
        
        prob_selected, rewards = run_episode(env, policy_model, device)
        loss = loss_fn(rewards, prob_selected, device=device, gamma=0.99)

        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode+1}  |  Loss {loss.item():.3f}  |  Total reward {sum(rewards):.2f}")
    
    return None

if __name__ == "__main__":
    
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)

    print("env done")

    policy_model = build_policy_model()

    print("model done")

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=5e-6)

    train(env, policy_model, optimizer, n_episodes=10)

    torch.save(policy_model.state_dict(), "policy_model.pth")
    print("Model saved to policy_model.pth")

    env.close()

