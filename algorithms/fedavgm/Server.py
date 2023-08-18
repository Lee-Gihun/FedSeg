import numpy as np
import copy
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.measures import *
from algorithms.fedavgm.ClientTrainer import ClientTrainer
from algorithms.fedavgm.utils import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
        )

        m_weights = copy.deepcopy(self.server_model)
        self.m_flat = flatten_weights(m_weights)
        self.m_flat = np.zeros_like(self.m_flat)
        self.avgm_beta = algo_params.avgm_beta

        print("\n>>> FedAvgM Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):

            # Initial Model Statistics
            if round_idx == 0:
                global_results = evaluate_model_on_loaders(
                    self.server_model, self.global_loaders, self.device
                )
                self.server_results = self._results_updater(
                    self.server_results, global_results
                )

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)

            # (Distributed) global weights
            dg_weights = copy.deepcopy(self.server_model.state_dict())
            if round_idx == 0:
                self.m_flat = flatten_weights(dg_weights, from_dict=True)

            # Client training stage to upload weights & stats
            (
                updated_local_weights,
                client_sizes,
                round_local_results,
            ) = self._clients_training(sampled_clients)

            # Get aggregated weights & update global (with server momemtum)
            new_weights = self._aggregation(updated_local_weights, client_sizes)
            self.server_model.load_state_dict(new_weights)

            dg_flat, new_flat = (
                flatten_weights(dg_weights, from_dict=True),
                flatten_weights(new_weights, from_dict=True),
            )

            server_momentum, ag_flat = update_momentum_weights(
                self.m_flat, dg_flat, new_flat, self.avgm_beta
            )
            self.m_flat = server_momentum

            # Update Global Server Model
            assign_weights(self.server_model, ag_flat)

            # Evaluate server statistics
            global_results = evaluate_model_on_loaders(
                self.server_model, self.global_loaders, self.device, "Global"
            )
            self.server_results = self._results_updater(
                self.server_results, global_results
            )

            # Print results and logging
            print(
                f"\n[Round {round_idx+1}/{self.n_rounds}] (Elapsed {round(time.time()-start_time, 1)}s)"
            )
            self._print_stats_with_logging(global_results, round_idx)
            self._print_stats_with_logging(round_local_results, round_idx)

            # Change learning rate
            if self.scheduler is not None:
                self.scheduler.step()
