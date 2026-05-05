from veros.settings import Setting

SETTINGS = {
    "start_date": Setting("",str, "Start date of simulation"),
    "dt_forcing": Setting(3, int, "Time step for forcing in hours"),
    "smooth": Setting(2, int, "scale ratio for topography gaussian filtering"),
    "baseline_model": Setting(False, bool,"no nn correction, run the biased model"),
    "hidden_dim_nn": Setting(128, int),
    "depth_nn": Setting(3, int),
    "nb_epochs": Setting(200, int),
    "batch_size": Setting(1, int),
    "nb_days_training": Setting(3, int),
    "lr": Setting(1e-3, float),
    "train_epsilon": Setting(False, bool),
    "epsilon": Setting(1e-6, float),
    "viz_mlp_freq": Setting(60*60*24*360, int),
    "normalize_inputs": Setting(True, bool),
    "normalize_loss": Setting(True, bool),
    "scaling_method_features": Setting("minmax",str),
    "scaling_method_loss": Setting("minmax",str),
    "use_lon_lat": Setting(False, bool,"use longitude and latitude as inputs of the MLP"),
    "conservative_mixing": Setting(True, bool,"use one hot encoding for features of MLP when multiple features per mlp"),
    "use_one_hot": Setting(False, bool),
    "seed": Setting(0, int),
    "num_mlps": Setting(1,int,"Number of MLPs to use: 1, 2 or 4"),
    "loss_type": Setting("zonal_mean", str,"Type of loss function to use. Options: 'zonal_mean' or 'pointwise'. Default: 'pointwise'"),
    "TS": Setting(True, bool,"If set, use only T,S as inputs but still predict T,S,U,V"),
    "reweight_loss": Setting(True, bool),
    "keys_loss": Setting(['temp','salt', 'ssh'], list,"List of variable names to include in the loss computation"),
    "resume_training": Setting(False, bool,"Resume training from existing checkpoint"),
    "resume_checkpoint_path": Setting("",str),
    "overwrite_training": Setting(False, bool)
    }

