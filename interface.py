from utils  import *
from painter import *
def train_data(model, data_name="pbmc", z_dim=50, alpha=0.1,
                               n_epochs=1000,
                               batch_size=32,
                               dropout_rate=0.25,
                               learning_rate=0.001,
                               condition_key="condition"):
    stim_key ,ctrl_key ,cell_type_key = get_key(data_name)
    train = sc.read(url+f"train_{data_name}.h5ad")
    valid = sc.read(url+f"valid_{data_name}.h5ad")

    for cell_type in train.obs[cell_type_key].unique().tolist():
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        net_valid_data = valid[~((valid.obs[cell_type_key] == cell_type) & (valid.obs[condition_key] == stim_key))]
        network =model(x_dimension=net_train_data.X.shape[1],
                                 z_dimension=z_dim,
                                 alpha=alpha,
                                 dropout_rate=dropout_rate,
                                 learning_rate=learning_rate,
                                 model_path=url2+f"scGen/{data_name}/{cell_type}/")

        network.train_vae(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs,
                      batch_size=batch_size)
        print(f"network_{cell_type} has been trained!")

# Predict
def reconstruct_data(model, data_name="pbmc", condition_key="condition"):
    stim_key ,ctrl_key ,cell_type_key = get_key(data_name) # Hpoly.Day10 Control cell_label
    train = sc.read(url+f"train_{data_name}.h5ad")
    all_data = anndata.AnnData()
    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        print(f"Reconstructing for {cell_type}")
        network = model(x_dimension=train.X.shape[1],
                                  z_dimension=100,
                                  alpha=0.00005,
                                  dropout_rate=0.2,
                                  learning_rate=0.001,
                                  model_path=url2+f"scGen/{data_name}/{cell_type}/")
        network.restore_model()

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]

        pred, delta = network.predict(adata=net_train_data,
                                      conditions={"ctrl": ctrl_key, "stim": stim_key},
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)
        pred=pred.detach().cpu().numpy()
        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_stim"] * len(pred),
                                                cell_type_key: [cell_type] * len(pred)},
                                      var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                      obs={condition_key: [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                          cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                      var={"var_names": cell_type_ctrl_data.var_names})
        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={condition_key: [f"{cell_type}_real_stim"] * len(real_stim),
                                                cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        print(f"Finish Reconstructing for {cell_type}")
    reconstructed_url = url2+f"reconstructed/"
    if not os.path.exists(reconstructed_url):
        os.makedirs(reconstructed_url)
    all_data.write_h5ad(reconstructed_url+f"{data_name}.h5ad")

import argparse
import importlib
def main():
    parser = argparse.ArgumentParser(description="ScDDPM")

    parser.add_argument('--num_epochs', type=int, default=10, help='Training epoch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type')
    parser.add_argument('--save_path', type=str, required=True, help='Saving path')
    parser.add_argument('--network', type=str, required=True, help='Network model')

    args = parser.parse_args()

    model_name= args.network
    package_name = "model"
    model_names = os.listdir(package_name)
    module_object = importlib.import_module(f"{package_name}.{model_name}")
    module_object_cls = getattr(module_object, model_name)  # 获取类对象
    model = module_object_cls()

    url = args.dataset_path
    url2 = args.saving_path
    using = args.dataset_type
    n_epochs = args.num_epochs
    train_data_path = sc.read(url + f"train_{using}.h5ad")
    reconstructed_data_path = sc.read(url2 + f"reconstructed/{using}.h5ad")

    train_data(model, data_name=using, z_dim=100, alpha=0.00005, n_epochs=n_epochs, batch_size=32,
               dropout_rate=0.2, learning_rate=0.01)

    reconstruct_data(model, using)

    hpoly = sc.read(train_data_path)
    hpoly_reconstructed = sc.read(reconstructed_data_path)

    cell_type = "TA.Early"
    conditions = {"ctrl": f"{cell_type}_ctrl", "pred_stim": f"{cell_type}_pred_stim",
                  "real_stim": f"{cell_type}_real_stim"}
    hpoly_cell = hpoly[hpoly.obs["cell_label"] == cell_type]
    sc.tl.rank_genes_groups(hpoly_cell, groupby="condition", n_genes=100, method="wilcoxon")
    diff_genes = hpoly_cell.uns["rank_genes_groups"]["names"]["Hpoly.Day10"].tolist()[:50] \
                 + hpoly_cell.uns["rank_genes_groups"]["names"]["Control"].tolist()[:50]
    path_to_save = f"./"
    reg_mean_plot(hpoly_reconstructed,
                  labels={"x": "", "y": ""},
                  condition_key="condition",
                  axis_keys={"x": conditions["pred_stim"], "y": conditions["real_stim"]},
                  gene_list=diff_genes[:5] + diff_genes[50:55],
                  top_100_genes=diff_genes,
                  path_to_save=os.path.join(path_to_save, f"Fig3a_hpoly_reg_mean.pdf"),
                  legend=False,
                  fontsize=18,
                  textsize=14,
                  x_coeff=0.35,
                  title="",
                  show=True)

if __name__ == "__main__":
    main()
