import torch
import torch.nn as nn
import os.path as osp


class DemoBlindAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_embed = nn.Linear(3, 32)
        self.prev_action_embedding = nn.Embedding(5, 32)
        self.lstm = nn.LSTM(64, 512, 2)
        self.fc = nn.Linear(512, 4)

        self.reset()

        self.load_state_dict(
            torch.load(osp.join(osp.dirname(__file__), "blind_agent_state.pth"))
        )

    def reset(self):
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(2, 1, 512, device=device),
            torch.zeros(2, 1, 512, device=device),
        )
        self.prev_action = torch.full((1,), -1, dtype=torch.long, device=device)

    def forward(self, obs):
        if isinstance(obs, dict):
            pg = obs["pointgoal"]
        else:
            pg = obs

        if not torch.is_tensor(pg):
            pg = torch.tensor(pg).to(
                device=next(self.parameters()).device, dtype=torch.float32
            )

        if len(pg.size()) == 1:
            pg = pg.unsqueeze(0)

        with torch.no_grad():
            pg = torch.stack([pg[:, 0], torch.cos(-pg[:, 1]), torch.sin(-pg[:, 1])], -1)

            x = torch.cat(
                [self.tgt_embed(pg), self.prev_action_embedding(self.prev_action + 1)],
                -1,
            )
            x = x.unsqueeze(0)

            x, self.hidden = self.lstm(x, self.hidden)

            x = self.fc(x)
            dist = torch.distributions.Categorical(logits=x)

            self.prev_action = dist.sample().squeeze(-1)

        return self.prev_action[0].item()


if __name__ == "__main__":
    obs = torch.randn(2)
    blind_agent = DemoBlindAgent()
    print(blind_agent(obs))
    print(blind_agent(obs))
