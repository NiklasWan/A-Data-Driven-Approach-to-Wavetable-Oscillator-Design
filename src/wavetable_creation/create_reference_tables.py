import argparse
import glob
import os
import scipy.io.wavfile as wav_tools
import audio_slice
import soundfile as sf
import wavetable_synthesis as wt_synth

INFO_TEXT: str = "This is the command line application for creating wavetable training data for TableGAN model."
WT_SIZE = 4096


def setup_argparse():
    parser = argparse.ArgumentParser(description=INFO_TEXT)

    parser.add_argument("-i", "--input", help="set the input directory path where "
                                              "audio files are residing", type=str)
    parser.add_argument("-o", "--output", help="set the output directory path where "
                                               "wavetables should be saved", type=str)
    parser.add_argument("-s", "--smoothing", help="true/false set if tan() smoothing "
                                                  "should be applied to waveform edges")
    parser.add_argument("-m", "--max_tables", help="set the maximum number of tables "
                                                   "which should be created for one audio file", type=int, default=64)
    parser.add_argument("-f", "--fast_wav", help="set whether input files are encoded 16-Bit or 32-Bit PCM = true"
                                                 "or other PCM encodings = false")

    return parser


def decode_audio(fp, fast_wav):
    audio_dict = dict(name=[], data=[], rate=[])

    for f in fp:
        if fast_wav:
            rate, data = wav_tools.read(f)
        else:
            data, rate = sf.read(f)
        data = data[:,0]
        audio_dict['name'].append(os.path.basename(f))
        audio_dict['data'].append(data)
        audio_dict['rate'].append(rate)

    return audio_dict


def make_filenames(num, name):
    names = []
    parts = name.split('.')

    for i in range(num):
        names.append(parts[0] + f'_{i}.' + parts[1])

    return names


def extract_cycles(audio_dict, smoothing, max_tables):

    cycles = dict(name=[], data=[])
    names = audio_dict['name']
    data = audio_dict['data']
    rates = audio_dict['rate']

    length = len(names)

    for i, name in enumerate(names):
        print(f'Slicing {(i+1)/length * 100:.2f} %')
        audio = data[i]
        rate = rates[i]

        slicer = audio_slice.AudioSlicer(audio, rate, WT_SIZE, smoothing, max_tables)
        slices = slicer.slice()
        num = len(slices)
        names = make_filenames(num, name)
        cycles['name'].append(names)
        cycles['data'].append(slices)

    return cycles


def save_to_disk(table_dict, file_path):
    path = os.path.join(file_path, 'output')

    if not os.path.exists(path):
        os.makedirs(path)

    names = table_dict['name']
    data = table_dict['data']

    length = len(names)

    for i, wt_names in enumerate(names):
        print(f'Saving {(i + 1) / length * 100:.2f} %')
        for j, name in enumerate(wt_names):
            save_path = os.path.join(path, name)
            wav_tools.write(save_path, 44100, data[i][j])


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


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    smoothing = "True"
    max_tables = 64
    fast_wav = "True"

    if args.input is None or args.input == "" or not os.path.isdir(args.input):
        raise Exception('You have to set a valid input path!')

    if args.output is None or args.output == "" or not os.path.isdir(args.output):
        raise Exception('You have to set a valid output path!')

    smoothing = args.smoothing.lower() if args.smoothing is not None else smoothing.lower()

    if smoothing != "false" and smoothing != "true":
        raise Exception('You have to set a valid value for smoothing!')
    else:
        smoothing = True if smoothing == "true" else False

    fast_wav = args.fast_wav.lower() if args.fast_wav is not None else fast_wav.lower()

    if fast_wav != "false" and fast_wav != "true":
        raise Exception('You have to set a valid value for smoothing!')
    else:
        fast_wav = True if fast_wav == "true" else False

    max_tables = args.max_tables
    input_path = args.input
    output_path = args.output

    fp = glob.glob(os.path.join(input_path, "*"))

    if len(fp) == 0:
        raise Exception("No files in input directory")

    audio_dict = decode_audio(fp, fast_wav)

    table_dict = extract_cycles(audio_dict, smoothing, max_tables)

    create_synth_tables_and_save(table_dict, output_path)


if __name__ == '__main__':
    main()
