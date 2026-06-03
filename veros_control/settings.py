from veros.settings import Setting

SETTINGS = {
    "start_date": Setting("",str, "Start date of simulation"),
    "dt_forcing": Setting(3, int, "Time step for forcing in hours"),
    "smooth": Setting(2, int, "scale ratio for topography gaussian filtering"),
    "baseline_model": Setting(False, bool, "no nn correction, run the biased model"),
    "hidden_dim_nn": Setting(128, int, "dimension of hidden layer"),
    "depth_nn": Setting(3, int, "number of hidden layer"),
    "nb_epochs": Setting(200, int, "number of epochs"),
    "batch_size": Setting(1, int, "batch size"),
    "nb_days_training": Setting(3, int, "number of days per training sequence"),
    "lr": Setting(1e-3, float, "learning rate"),
    "train_epsilon": Setting(False, bool, "if True, train espilon parameter"),
    "epsilon": Setting(1e-6, float,"epsilon default value"),
    "viz_mlp_freq": Setting(60*60*24*360, int, "frequence of visualization"),
    "normalize_inputs": Setting(True, bool, "if True, normalize inputs"),
    "normalize_loss": Setting(True, bool, "if True, normalize loss"),
    "scaling_method_features": Setting("minmax",str,"scaling method for features"),
    "scaling_method_loss": Setting("minmax",str, "scaling methos for loss"),
    "use_lon_lat": Setting(False, bool,"use longitude and latitude as inputs of the MLP"),
    "conservative_mixing": Setting(True, bool,"use one hot encoding for features of MLP when multiple features per mlp"),
    "use_one_hot": Setting(False, bool, "if True, add one hot layer"),
    "seed": Setting(0, int, "random seed"),
    "num_mlps": Setting(1,int,"Number of MLPs to use: 1, 2 or 4"),
    "loss_type": Setting("zonal_mean", str,"Type of loss function to use. Options: 'zonal_mean' or 'pointwise'. Default: 'pointwise'"),
    "TS": Setting(True, bool,"If set, use only T,S as inputs but still predict T,S,U,V"),
    "reweight_loss": Setting(True, bool,"if True, reweight loss"),
    #"keys_loss": Setting(['temp','salt', 'ssh'], list,"List of variable names to include in the loss computation"),
    "resume_training": Setting(False, bool, "Resume training from existing checkpoint"),
    "resume_checkpoint_path": Setting("",str, "Resume training from existing checkpoint with path"),
    "overwrite_training": Setting(False, bool, "refresh training")
    }

