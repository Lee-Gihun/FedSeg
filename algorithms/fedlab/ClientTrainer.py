import torch
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

        self.rgb_means = {
            0: [0.475, 0.426, 0.346],
            1: [0.585, 0.476, 0.296],
            2: [0.495, 0.466, 0.316],
            3: [0.285, 0.356, 0.306],
        }

        self.rgb_stds = {
            0: [0.239, 0.234, 0.235],
            1: [0.129, 0.324, 0.135],
            2: [0.249, 0.214, 0.425],
            3: [0.269, 0.234, 0.125],
        }

    def train(self, client_idx=None):
        """Local training"""

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize
        mu_s = torch.Tensor(self.rgb_means[client_idx]).view(1, 3, 1, 1).to(self.device)
        sigma_s = (
            torch.Tensor(self.rgb_stds[client_idx]).view(1, 3, 1, 1).to(self.device)
        )

        a = random.choice([x for x in range(4) if x != client_idx])
        mu_t = torch.Tensor(self.rgb_means[a]).view(1, 3, 1, 1).to(self.device)
        sigma_t = torch.Tensor(self.rgb_means[a]).view(1, 3, 1, 1).to(self.device)

        for _ in range(self.local_epochs):
            for data, targets, _ in self.local_loaders["train"]:
                self.optimizer.zero_grad()

                # forward pass

                data, targets = data.to(self.device), targets.to(self.device)
                data = ((data - mu_s) / sigma_s) * sigma_t + mu_t
                output = self.model(data)
                loss = self.criterion(output, targets.long())

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size
