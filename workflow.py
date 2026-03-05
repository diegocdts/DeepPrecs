import argparse

from deepprecs4deblending import workflow_deblending

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=False, help="Data name")
    parser.add_argument("--train_model", type=str, required=False, help="To train or not to train the model")

    args = parser.parse_args()

    label = args.label
    train_model = True if args.train_model is None or args.train_model.lower() == 'true' else False

    if label == '5D' or label == '5_D':
        label = '5D'
        inputfile = '/home/data/5D/Streamer/05_D_clean.sgy'
        ns = 256
    elif label == '6A' or label == '6_A':
        label = '6A'
        inputfile = '/home/data/6A/Streamer/06_A_clean.sgy'
        ns = 1666
    elif label == 'marmousi':
        inputfile = '/home/data/Marmousi/Streamer/Marmousi_clean.sgy'
        ns = 120
    elif label == 'seam':
        inputfile = '/home/data/Seam/Streamer/Seam_clean.sgy'
        ns = 1200
    else:
        raise ValueError("Dataset not found!")

    workflow_deblending(label, inputfile, ns, train_model)