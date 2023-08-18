import os
from .acdc.loader import get_dataloader_acdc


__all__ = ["data_distributer"]


def data_distributer(root, dataset_name, batch_size, n_clients, localonlyid=None):
    root = os.path.join(root, dataset_name)

    print(">>> Distributing Client Data...")

    local_loaders, local_sizes = {}, {}

    # Centralized Setup
    if n_clients == 1:
        print(f">>> Building Centralized data...")
        local_trainloader = get_dataloader_acdc(
            root, train=True, client_id=localonlyid, batch_size=batch_size,
        )
        local_loaders[0] = {
            "train": local_trainloader,
        }
        local_sizes[0] = 400

    else:
        for client_idx in range(n_clients):
            print(f">>> Building Client {client_idx} data...")
            local_trainloader = get_dataloader_acdc(
                root,
                train=True,
                batch_size=batch_size,
                client_id=client_idx,
                out_client=False,
            )
            local_test_in_loader = get_dataloader_acdc(
                root,
                train=False,
                batch_size=batch_size,
                client_id=client_idx,
                out_client=False,
            )
            local_test_out_loader = get_dataloader_acdc(
                root,
                train=False,
                batch_size=batch_size,
                client_id=client_idx,
                out_client=True,
            )

            local_loaders[client_idx] = {
                "train": local_trainloader,
                "test_in": local_test_in_loader,
                "test_out": local_test_out_loader,
            }
            local_sizes[client_idx] = 100

    print(">>> Building Global evaluation data...")
    global_loaders = {}

    client_domains = ["rain", "fog", "snow", "night"]
    for client_idx in range(4):
        test_loader = get_dataloader_acdc(
            root,
            train=False,
            batch_size=batch_size,
            client_id=client_idx,
            out_client=False,
        )
        global_loaders[client_domains[client_idx]] = test_loader

    data_distributed = {
        "local": local_loaders,
        "global": global_loaders,
        "local_sizes": local_sizes,
    }

    return data_distributed
