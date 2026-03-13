
import warnings
warnings.filterwarnings('ignore')

import os
import segyio
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pylops

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.sparse import csr_matrix, vstack
from scipy.signal import filtfilt, convolve
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import LinearOperator, cg, lsqr
from scipy import misc
from torchsummary import summary
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping as pl_EarlyStopping

from pylops.utils import dottest
from pylops.utils.wavelets import *
from pylops.utils.tapers import *
from pylops.basicoperators import *
from pylops.signalprocessing import *
from pylops.waveeqprocessing.oneway import PhaseShift, Deghosting
from pylops.optimization.sparsity  import *
from pylops_gpu import TorchOperator

from deepprecs.deblend import BlendingContinuous
from deepprecs.patching import patching, patch_scalings, patch2d_design
from deepprecs.subsampling import subsampling 
from deepprecs.aemodel import AutoencoderBase, AutoencoderRes, AutoencoderMultiRes
from deepprecs.train_pl import *
from deepprecs.invert import InvertAll


def workflow_deblending(label, inputfile, ns, train_model):
    ################# GLOBAL ####################
    # Device
    devicenum = 0
    device = torch.device(f"cuda:{devicenum}" if torch.cuda.is_available() else "cpu")
    print(device)
    if 'cuda' in str(device):
        import cupy as cp

    # Seed
    seed_everything(5)

    # Model and figure directories
    outputmodels = f'results/{label}/models'
    outputfigs = f'results/{label}/figures'

    os.makedirs(outputmodels, exist_ok=True)
    os.makedirs(outputfigs, exist_ok=True)

    ################# TRAINING ####################

    # Patches
    nspatch, ntpatch = 64, 64
    nsjump, ntjump = 32, 32

    # Experiment number and name
    iexp = 0
    expname = 'ResNet network with more filters 300enc - mse+ccc cost with learned weigths + mask'

    # AE parameters
    aetype = AutoencoderRes # AE type: AutoencoderBase, AutoencoderRes, AutoencoderMultiRes
    nenc = 300 # Size of latent space
    kernel_size = 5 # Size of filters kernels
    nfilts = 32 # number of filters for conv layers in first level (doubles up going down)
    nlayers = 2 # number of layers per level 
    nlevels = 2 # number of levels
    convbias = True # add bias to convolution layers 
    downstride = 1 # stride of downsampling/pooling blocks (same will be used for upsampling blocks)
    downmode = 'max' # type of pooling (avg or max)
    upmode = 'upsample' # type of upsampling (convtransp, upsample or upsample1d)
    bnormlast = True # add batch normalization to the last layer
    act_fun = 'LeakyReLU' # activation function for all hidden layers
    relu_enc = False # add ReLU activation to the linear layer of the encoder (this and tanh_enc cannot be both true)
    tanh_enc = True # add TanH activation to the linear layer of the encoder
    relu_dec = True # add ReLU activation to the linear layer of the decoder
    tanh_final = False # add TanH activation to the last layer of the network - ensures the output is bounded between [-1, 1]

    # Loss/optimizer
    loss = 'mse_ccc' # loss: mse, weightmse, l1, ssim, peaerson, mse_pearson, l1_pearson, ccc, mse_ccc, l1_ccc
    lossweights = None
    betas = (0.5, 0.9) # betas of Adam optimizer
    weight_decay = 1e-5 # weigth decay of Adam optimizer
    learning_rate = 1e-4 # learning rate of Adam optimizer
    adapt_learning = True # apply adaptive learning rate
    lr_scheduler = 'OneCycle' # type of adaptive learning rate: OnPlateau, OneCycleLR
    lr_factor = None # lr factor for OnPlateau
    lr_thresh = None # lr thresh for OnPlateau
    lr_patience = None # lr patience for OnPlateau
    lr_max = 1e-3 # lr max for OneCycleLR
    es_patience = 10 # early stopping patience
    es_min_delta = 1e-3 # min difference to trigger early stopping

    # Training
    num_epochs = 1 # number of epochs
    batch_size = 256 # batch size
    noise_std = 0.0 # standard deviation noise to input
    mask_perc = 0.2 # percentage of traces to mask

    ####### DEBLENDING #######
    # Receiver index to be used for inversion
    irec = 100

    # Patching operator
    nwin = (nspatch, ntpatch)
    nop = (nspatch, ntpatch)
    nover = (19, 8)

    ################# LOAD DATA ####################

    with segyio.open(inputfile, "r", ignore_geometry=True) as f:

        ntraces = f.tracecount
        nt = len(f.samples)

        # tempo
        dt = segyio.tools.dt(f) / 1e6
        t = np.arange(nt) * dt

        # dados
        data = segyio.tools.collect(f.trace[:])   # (ntraces, nt)

        # headers
        sx = f.attributes(segyio.TraceField.FieldRecord)[:]
        gx = f.attributes(segyio.TraceField.TraceNumber)[:]

    s_unique, s_index = np.unique(sx, return_inverse=True)
    r_unique, r_index = np.unique(gx, return_inverse=True)

    if len(s_unique) == 1:
        nr = int(len(r_unique) / ns)

        sx = np.arange(ns)
        gx = np.arange(nr)

        s_unique, s_index = np.unique(sx, return_inverse=True)
        r_unique, r_index = np.unique(gx, return_inverse=True)

    ds = np.median(np.diff(s_unique))
    dr = np.median(np.diff(r_unique))

    s = s_unique.reshape(1, ns)
    r = r_unique.reshape(1, nr)

    p = data.reshape(ns, nr, nt)

    p /= np.max(np.abs(p))

    print("===== Informações do dado sísmico =====")

    print(f"ns (número de sources)    : {ns}")
    print(f"nr (número de receivers)  : {nr}")

    print(f"ds (espaçamento sources)  : {ds}")
    print(f"dr (espaçamento receivers): {dr}")

    print(f"nt (número de amostras)   : {nt}")
    print(f"dt (passo temporal)       : {dt}")

    print("\nShapes das variáveis:")
    print(f"s shape: {s.shape}")
    print(f"r shape: {r.shape}")
    print(f"t shape: {t.shape}")
    print(f"p shape: {p.shape}")

    print("\nIntervalos:")
    print(f"s min/max: {s.min()}  {s.max()}")
    print(f"r min/max: {r.min()}  {r.max()}")
    print(f"t min/max: {t.min()}  {t.max()}")

    ################# DEBLENDING OPERATOR ####################

    # Blending operator
    overlap = 0.33
    jitter = np.random.uniform(-.15, .15, ns)
    ignition_times = np.arange(0, overlap * nt * ns, overlap * nt) * dt + jitter
    ignition_times[0] = 0.0

    # Blending
    Bop = BlendingContinuous(nt, nr, ns, dt, ignition_times.astype("float32"), dtype=np.float32)
    dottest(Bop, verb=True, tol=1e-2)
    pblended = Bop * p.ravel()
    ppseudo = Bop.H * pblended

    pblended = pblended.reshape(Bop.nr, Bop.nttot)
    ppseudo = ppseudo.reshape(Bop.ns, Bop.nr, Bop.nt)

    # Create data to deblend (single receiver-gather)
    pblend = pblended[irec] 

    # Convert data to torch
    pblend_torch = torch.from_numpy(pblend.astype(np.float32)).to(device)

    # Create blending operator for single receiver
    B1op = BlendingContinuous(nt, 1, ns, dt, ignition_times.astype("float32"), dtype=np.float32)

    ppseudo1 = B1op.H * pblend[np.newaxis].ravel()
    ppseudo1 = ppseudo1.reshape(ns, nt)

    # Patch operator
    dimsd = (ns, nt)
    npatches = patch2d_design(dimsd, (nspatch, ntpatch), nover, nop)[0]
    dims = (npatches[0]*nspatch, npatches[1]*ntpatch)

    Op = Identity(nspatch*ntpatch, dtype='float32')
    Pop = Patch2D(Op, dims, dimsd, nwin, nover, nop,
                tapertype=None, design=True)
    Pop1 = Patch2D(Op.H, dims, dimsd, nwin, nover, nop,
                tapertype='cosine', design=False)
    Pop1_torch = TorchOperator(Pop1, pylops=True, device=device)

    # Find scalings
    scalings = patch_scalings(ppseudo1, Pop, npatches, npatch=(nspatch, ntpatch), 
                            plotflag=True, clip=clip, device=device)

    # Overall operator
    B1op_torch = TorchOperator(B1op, pylops=True, device=device)
    
    ################# DATASET GENERATION ####################

    # Create patches
    xs = patching(pblended[:, np.newaxis, :], s, r, dt, 
                npatch=(nspatch, ntpatch), njump=(nsjump, ntjump), window=False, 
                augumentdirect=False)
    
    ################# TRAINING ####################

    print('Working with Experiment %d - %s' % (iexp, expname))

    # Create directory to save training evolution
    figdir = os.path.join(outputfigs, 'exp%d' % iexp)

    if figdir is not None:
        if not os.path.exists(figdir):
            os.mkdir(figdir)

    # Create model to train
    autoencoder = aetype(nh=nspatch, nw=ntpatch, nenc=nenc, 
                        kernel_size=kernel_size, nfilts=nfilts, 
                        nlayers=nlayers, nlevels=nlevels,
                        physics=B1op_torch, 
                        convbias=convbias, act_fun=act_fun, 
                        downstride=downstride, downmode=downmode,
                        upmode=upmode, bnormlast=bnormlast,  
                        relu_enc=relu_enc, tanh_enc=tanh_enc, 
                        relu_dec=relu_dec, tanh_final=tanh_final,
                        patcher=Pop1_torch, npatches=npatches[0]*npatches[1],
                        patchesscaling=scalings)

    # Create dataset
    datamodule = DataModule(xs, valid_size=0.1, random_state=42, batch_size=batch_size)

    # Callbacks
    early_stop_callback = pl_EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=es_patience,
                                        min_delta=es_min_delta)
    callback = MetricsCallback(loss)
    callback1 = PlottingCallback(figdir)

    # Training
    if train_model:
        dimred = LitAutoencoder(nspatch, ntpatch, nenc,
                                autoencoder, loss, num_epochs, lossweights=lossweights,
                                learning_rate=learning_rate, weight_decay=weight_decay, betas=betas,
                                adapt_learning=True, lr_scheduler=lr_scheduler, lr_factor=lr_factor,
                                lr_thresh=lr_thresh, lr_patience=lr_patience, lr_max=lr_max,
                                noise_std=noise_std, mask_perc=mask_perc, device=device)

        trainer = pl.Trainer(accelerator='gpu', devices=[devicenum, ],
                            max_epochs=dimred.num_epochs, log_every_n_steps=4, 
                            callbacks=[early_stop_callback, callback, callback1])
        trainer.fit(dimred, datamodule)

        # Save model
        torch.save(autoencoder.state_dict(), os.path.join(outputmodels, 'exp%d_modelweights.pt' % iexp))
    else:
        model_path = os.path.join(outputmodels, f'exp{iexp}_modelweights.pt')
        autoencoder.load_state_dict(torch.load(model_path, map_location=device))
        autoencoder.eval()

    ################# DEBLENDING ####################

    print('Deblending...')

    # Initial guess
    patchesmask = Pop.H * ppseudo1.ravel()
    patchesmask = patchesmask.reshape(npatches[0]*npatches[1], 1, nspatch, ntpatch)
    patchesmask_scaled = patchesmask / scalings.cpu().detach().numpy()
    p0 = autoencoder.encode(torch.from_numpy(patchesmask_scaled.astype(np.float32)).to(device)).cpu().detach().numpy()

    # Invert
    inv = InvertAll(device, # device
                    nenc, npatches[0] * npatches[1],
                    autoencoder, autoencoder.patched_physics_decode, autoencoder.patched_decode, # modelling ops
                    nn.MSELoss(), 1., 100, # optimizer
                    reg_ae=0., x0=p0, bounds=None
                    )
    
    for i_rec in range(nr):
        # Create data to deblend (single receiver-gather)
        pblend = pblended[i_rec] 

        # Convert data to torch
        pblend_torch = torch.from_numpy(pblend.astype(np.float32)).to(device)

        minv, pinv = inv.scipy_invert(pblend_torch, torch.zeros(((ns, nt))).to(device))

        # Recompute data from minv
        if 'cuda' in str(device):
            dinv = cp.asnumpy(B1op * cp.asarray(minv))
        else:
            dinv = Dupop * minv

        minv = minv.reshape(ns, nt)
        dinv = dinv.reshape(Bop.nttot)

        minv_refined = cp.asnumpy(pylops.optimization.solver.lsqr(B1op, cp.asarray(pblend), 
                                                         cp.asarray(minv.ravel()), niter=10)[0]).reshape(ns, nt)
    
    print("End of processing")