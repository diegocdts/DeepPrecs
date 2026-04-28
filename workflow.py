import argparse

from deepprecs4deblending_norm_max_abs import workflow_deblending

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=False, help="Data name")
    parser.add_argument("--train_model", type=str, required=False, help="To train or not to train the model")

    args = parser.parse_args()

    label = args.label
    train_model = True if args.train_model is None or args.train_model.lower() == 'true' else False

    if '5D' in label:
        label = '5D'
        inputfile = '/home/data/05_D_blend.npy'
        ns = 256
    elif '6A' in label:
        label = '6A'
        inputfile = '/home/data/06_A_.npy'
        ns = 1666
    elif 'marmousi' in label:
        inputfile = '/home/data/marmousi_blend.npy'
        ignition_times = '/home/data/marmousi_times.npy'
        ns = 1001
        nr = 120
        nt = 1500
        dt = 0.004
    elif 'seam' in label:
        inputfile = '/home/data/seam_blend.npy'
        ns = 1200
    else:
        raise ValueError("Dataset not found!")

    workflow_deblending(label, inputfile, ignition_times, ns, nr, nt, dt, train_model)