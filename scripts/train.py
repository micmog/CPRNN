from pathlib import Path

import torch
import torch.nn.functional as Func
from torch.utils.data import DataLoader, random_split

from cprnn.data import load_dataset_matflow
from cprnn.model import CP_RNN


device = torch.device("cuda")
# device = torch.device('cpu')
train_frac = 0.01
seed = 42
data_path = Path("/home/ir-atki2/scratch")
wk_name = "generate_training_data_texture_2024-06-09_142205"
output_path = Path("./")

NEPOCHS = 1000000
load_state = False
learning_rate = 1.0e-5

OPATH = "output.txt"
SPATH = "s_lstm.pt"
SOPTP = "s_optim.pt"
SPATH_T = "s_lstm_test.pt"
SOPTP_T = "s_optim_test.pt"

print("start loading")
dataset, dataset_state = load_dataset_matflow(data_path / wk_name, 'voronoi')
print("done loading")

generator = torch.Generator().manual_seed(seed)
init_state = generator.get_state().clone()
train_dataset, test_dataset = random_split(
    dataset, [1 - train_frac, train_frac], generator=generator
)
generator.set_state(init_state)
train_dataset_state, test_dataset_state = random_split(
    dataset_state, [1 - train_frac, train_frac], generator=generator
)

train_data = next(iter(DataLoader(train_dataset, batch_size=10000))).to(device)
test_data = next(iter(DataLoader(test_dataset, batch_size=10000))).to(device)
train_state = next(iter(DataLoader(train_dataset_state, batch_size=10000))).to(device)
test_state = next(iter(DataLoader(test_dataset_state, batch_size=10000))).to(device)

if not load_state:
    loss_min, loss_t_min = 1.0e32, 1.0e32

model = CP_RNN(dataset_state.state_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if load_state:
    model.load_state_dict(torch.load(output_path / SPATH))
    optimizer.load_state_dict(torch.load(output_path / SOPTP))
for group in optimizer.param_groups:
    group["lr"] = learning_rate

f = open(output_path / OPATH, "w")

X, Y = torch.split(train_data, [model.s_in_size, model.s_ou_size], dim=-1)
X_t, Y_t = torch.split(test_data, [model.s_in_size, model.s_ou_size], dim=-1)
for epoch in range(NEPOCHS):
    # Train model
    model.train()
    model.zero_grad(set_to_none=True)
    Y_hat = model(X, train_state)
    loss = Func.mse_loss(Y_hat, Y)
    loss.backward()
    optimizer.step()

    # Evaluate model
    model.eval()
    with torch.no_grad():
        Y_t_hat = model(X_t, test_state)
        loss_t = Func.mse_loss(Y_t_hat, Y_t)

    reducing = bool(loss < loss_min)
    if reducing:
        loss_min = loss
        torch.save(model.state_dict(), output_path / SPATH)
        torch.save(optimizer.state_dict(), output_path / SOPTP)

    reducing_t = bool(loss_t < loss_t_min)
    if reducing_t:
        loss_t_min = loss_t
        torch.save(model.state_dict(), output_path / SPATH_T)
        torch.save(optimizer.state_dict(), output_path / SOPTP_T)

    f.write(
        f"{epoch:>9} {loss:.5e} {loss_min:.5e} {int(reducing)} "
        f"{loss_t:.5e} {loss_t_min:.5e} {int(reducing_t)}\n"
    )
    f.flush()
