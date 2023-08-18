import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.fedntd.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer
from algorithms.fedntd.criterion import *

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
        local_criterion = self._get_local_criterion(self.algo_params, 19)

        self.client = ClientTrainer(
            local_criterion,
            algo_params=algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
        )

        print("\n>>> FedNTD Server initialized...\n")
        
        
    def _get_local_criterion(self, algo_params, num_classes):
        tau = algo_params.tau
        beta = algo_params.beta

        criterion = NTD_Loss(num_classes, tau, beta)

        return criterion
