import argparse
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import wavetable_synthesis as wt_synth

INFO_TEXT: str = "This is the command line application for creating wavetable training data for TableGAN model."
WT_SIZE = 4096


def setup_argparse():
    parser = argparse.ArgumentParser(description=INFO_TEXT)

    parser.add_argument("-i", "--input_model", help="choose the input model",
                        choices=['batch_norm', 'n_batch_norm', 'phase_shuffling', 'n_batch_norm_200K'],type=str)
    parser.add_argument("-o", "--output", help="set the output directory path where "
                                               "wavetables should be saved", type=str)
    parser.add_argument("-n", "--num_tables", help="set the number of tables "
                                                   "which should be created for one interpolation process",
                        type=int, default=64)
    parser.add_argument("-w", "--wavetable_name", help="set the prefix name for wavetables",
                        type=str)

    return parser


def make_filenames(num, name):
    names = []

    for i in range(num):
        names.append(name + f'_{i}')

    return names


def generate_latent_points(latent_dim, n_samples):

    z_input = np.random.uniform(-1.,1., (n_samples, latent_dim)).astype('float32')
    return z_input


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def choose_random_wave(model, num):
    print(f'Computing {num}. wave')

    z = None

    while True:
        z = generate_latent_points(100, 1)
        X = model.predict(z)
        x = np.arange(4096)

        plt.plot(x, X[0, :, 0])
        plt.show()
        choice = input(f'Do you want this as your {num}. waveform? (Y/N)')

        if choice.lower() == 'y':
            break

    if z is None:
        raise Exception(f'Error in creating {num}. wave')

    return z


def interpolate_slerp(p1, p2, n_steps=64):
    ratios = np.linspace(0, 1, num=n_steps)

    vectors = list()

    for ratio in ratios:
        y = slerp(ratio, p1, p2)
        vectors.append(y)

    return np.asarray(vectors)


def interpolate_points(p1, p2, n_steps=64):

    ratios = np.linspace(0, 1, num=n_steps)

    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


def create_synth_tables_and_save(table_dict, file_path):
    path = os.path.join(file_path, 'output')

    if not os.path.exists(path):
        os.makedirs(path)

    names = table_dict['name']
    data = table_dict['data']

    length = len(names)

    for i, wt_names in enumerate(names):
        print(f'FFT Synthesis and Save: {(i + 1) / length * 100:.2f} %')
        for j, name in enumerate(wt_names):
            save_path = os.path.join(path, name.split('_')[0])

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            fft_synth = wt_synth.WavetableSynthesis(data[i][j])
            stack = fft_synth.synthesise_all_frames()
            name = name.split('.')[0] + '.json'
            stack.serialize_to_file(os.path.join(save_path, name))


def create_table_dict(preds, names):
    preds = preds[:, :, 0].tolist()
    dct = dict(name=[], data=[])
    dct['name'].append(names)
    dct['data'].append(preds)

    return dct


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    if args.input_model is None or args.input_model == "":
        raise Exception('You have to set a valid input model!')

    if args.output is None or args.output == "" or not os.path.isdir(args.output):
        raise Exception('You have to set a valid output path!')

    if args.wavetable_name == "":
        raise Exception('You have to give a valid prefix table name!')

    names = make_filenames(args.num_tables, args.wavetable_name)

    model = None

    if args.input_model == 'batch_norm':
        model = load_model('../trained_models/trained_generator_batch_norm.h5')
    elif args.input_model == 'n_batch_norm':
        model = load_model('../trained_models/trained_generator_no_batch_norm.h5')
    elif args.input_model == 'phase_shuffling':
        model = load_model('../trained_models/trained_generator_phase_shuffling.h5')
    elif args.input_model == 'n_batch_norm_200K':
        model = load_model('../trained_models/trained_generator_no_batch_norm200K.h5')

    z1 = choose_random_wave(model, 1)
    z1 = z1[0, :]
    z2 = choose_random_wave(model, 2)
    z2 = z2[0, :]
    interpolation = interpolate_slerp(z1, z2, args.num_tables)

    X = model.predict(interpolation)
   
    table_dict = create_table_dict(X, names)
    create_synth_tables_and_save(table_dict, args.output)


if __name__ == '__main__':
    main()