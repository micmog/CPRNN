from pathlib import Path

import torch
import torch.nn.functional as Func
from torch.utils.data import DataLoader, random_split

from cprnn.data import load_dataset_matflow
from cprnn.model import CP_RNN


device = torch.device("cuda")
# device = torch.device("mps")
# device = torch.device('cpu')
seed = 42
data_path = Path("/home/ir-atki2/scratch")
wk_name = "generate_training_data_texture_2024-06-09_142205"
# data_path = Path("/Users/michaelwhite/Documents/projects/cprnn/data")
# wk_name = "fingerprint_damask_hc"
output_path = Path("./")

config = {
        "n_epochs": 1000000,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "s_hs_size": 256,
        "s_n_layer": 3,
        "state_name": "fingerprint",
        "load_state": False,
        "batch_size": 10000,
        "train_frac": 0.01,
    }

OPATH = "output.txt"
SPATH = "s_lstm.pt"
SOPTP = "s_optim.pt"
SPATH_T = "s_lstm_test.pt"
SOPTP_T = "s_optim_test.pt"

print("start loading")
dataset, dataset_state = load_dataset_matflow(data_path / wk_name, config["state_name"])
print("done loading")

generator = torch.Generator().manual_seed(seed)
init_state = generator.get_state().clone()
train_dataset, test_dataset = random_split(
    dataset, [1 - config["train_frac"], config["train_frac"]], generator=generator
)
generator.set_state(init_state)
train_dataset_state, test_dataset_state = random_split(
    dataset_state, [1 - config["train_frac"], config["train_frac"]], generator=generator
)

train_data = DataLoader(train_dataset, batch_size=config["batch_size"])
test_data = DataLoader(test_dataset, batch_size=config["batch_size"])
train_state = DataLoader(train_dataset_state, batch_size=config["batch_size"])
test_state = DataLoader(test_dataset_state, batch_size=config["batch_size"])

if not config["load_state"]:
    loss_min, loss_t_min = 1.0e32, 1.0e32

model = CP_RNN(dataset_state.state_size, 0.0, config['s_hs_size'], config['s_n_layer']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
if config["load_state"]:
    model.load_state_dict(torch.load(output_path / SPATH))
    optimizer.load_state_dict(torch.load(output_path / SOPTP))
for group in optimizer.param_groups:
    group["lr"] = config["learning_rate"]

f = open(output_path / OPATH, "w")

for epoch in range(config["n_epochs"]):

    # Train model
    model.train()
    model.zero_grad(set_to_none=True)
    train_loss = 0.0
    n_batches = len(train_data)
    for _, (data_batch, state_batch) in enumerate(zip(train_data, train_state)):
        X, Y = torch.split(data_batch, [model.s_in_size, model.s_ou_size], dim=-1)
        Y_hat = model(X.to(device), state_batch.to(device))
        loss = Func.mse_loss(Y_hat, Y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= n_batches

    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        n_batches = len(test_data)
        for _, (data_t_batch, state_t_batch) in enumerate(zip(test_data, test_state)):
            X_t, Y_t = torch.split(data_t_batch, [model.s_in_size, model.s_ou_size], dim=-1)
            Y_t_hat = model(X_t.to(device), state_t_batch.to(device))
            loss_t = Func.mse_loss(Y_t_hat, Y_t.to(device))
            test_loss += loss_t.item()
        test_loss /= n_batches

    reducing = bool(loss < loss_min)
    if reducing:
        loss_min = train_loss
        torch.save(model.state_dict(), output_path / SPATH)
        torch.save(optimizer.state_dict(), output_path / SOPTP)

    reducing_t = bool(test_loss < loss_t_min)
    if reducing_t:
        loss_t_min = test_loss
        torch.save(model.state_dict(), output_path / SPATH_T)
        torch.save(optimizer.state_dict(), output_path / SOPTP_T)

    f.write(
        f"{epoch:>9} {loss:.5e} {loss_min:.5e} {int(reducing)} "
        f"{loss_t:.5e} {loss_t_min:.5e} {int(reducing_t)}\n"
    )
    f.flush()
