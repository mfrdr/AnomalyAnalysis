import os
import time

from tqdm import tqdm

from graphino.GCN.GCN_model_2 import GCN_2
# from eval_gcn import plot_adjacency_graph  # Import the plotting function

# Training settings
from graphino.training_2 import evaluate, train_epoch, get_dataloaders, get_dirs
from utilities.hyperparams_and_args_GCN import get_argparser
from utilities.utils import set_seed
from utilities.model_logging import update_tqdm, save_model
from utilities.optimization import get_optimizer, get_loss

params, net_params = get_argparser()
set_seed(params['seed'])
(adj_atm, adj_oc, static_feats_atm, static_feats_oc, _), (trainloader, valloader, testloader) = get_dataloaders(params, net_params)
ckpt_dir, log_dir = get_dirs(params, net_params)
# Model and optimizer
model = GCN_2(net_params, static_feat_atm=static_feats_atm, static_feat_oc=static_feats_oc, adj_atm=adj_atm, adj_oc=adj_oc)
optimizer = get_optimizer(params['optimizer'], model, lr=params['lr'], weight_decay=params['weight_decay'], nesterov=params['nesterov'])
criterion = get_loss(params['loss'])

# Train model
device = 'cuda'
t_total = time.time()
model = model.to(device)
val_stats = None
print('Params', params)
print('Net params', net_params)
with tqdm(range(1, params['epochs'] + 1)) as t:
    for epoch in t:
        t.set_description('Graphino')
        start_t = time.time()
        loss, num_edges = train_epoch(trainloader, model, criterion, optimizer, device, epoch)
        duration = time.time() - start_t

        if valloader is not None:
            _, val_stats = evaluate(valloader, model, device=device)
        _, test_stats = evaluate(testloader, model, device=device)
        update_tqdm(t, loss, n_edges=num_edges, time=duration, val_stats=val_stats, test_stats=test_stats)

        if epoch == 10:
            print('Saving model at epoch', epoch)
            save_model(model, ckpt_dir, params, net_params, optimizer, epoch, ID=f'{epoch}ep_2_model.pkl')

save_model(model, ckpt_dir, params, net_params, optimizer, epoch, ID='last_model.pkl')
print("Optimization Finished! Total time elapsed: {:.4f}s".format(time.time() - t_total))

