import wavetable
import json
import argparse
import os
import glob
import oscillator
import scipy.io.wavfile as wav_tools
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

INFO_TEXT: str = "This is the command line application for the wavetable oscillator."
WT_SIZE = 4096
SAMPLE_RATE = 44100

def setup_argparse():
    parser = argparse.ArgumentParser(description=INFO_TEXT)

    parser.add_argument("-i", "--input", help="choose the input path to wavetable json files", type=str)
    parser.add_argument("-o", "--output", help="set the output directory path where audio"
                                               " file should be saved", type=str)
    parser.add_argument("-p", "--pitch_automation", help="set to true if pitch should be automated",
                        type=str)
    parser.add_argument("-w", "--wt_automation", help="set to true if wavetable position should be automated",
                        type=str)
    parser.add_argument("-f", "--frequency", help="frequency at which the sound should be played back (20-20000 Hz)",
                        type=int, default=60)
    parser.add_argument("-l", "--playback_length", help="length in seconds for how long the sound should play",
                        type=float, default=8.0)

    return parser


def read_json_files(file_paths):
    import re
    stack_array = []

    list.sort(file_paths, key=lambda x: int(re.findall("_[0-9]+", x)[0].split('_')[1]))
    for f in file_paths:
        with open(f) as data_file:
            data_loaded = json.load(data_file)
            data_loaded = json.loads(data_loaded)

            table_stack = wavetable.WavetableStack.from_json_dict(data_loaded)
            stack_array.append(table_stack)

    return stack_array


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    pitch_automation = "False"
    wt_automation = "False"

    if args.input is None or args.input == "" or not os.path.isdir(args.input):
        raise Exception('You have to set a valid input path!')

    if args.output is None or args.output == "" or not os.path.isdir(args.output):
        raise Exception('You have to set a valid output path!')

    pitch_automation = args.pitch_automation.lower() if args.pitch_automation is not None else pitch_automation.lower()

    if pitch_automation != "false" and pitch_automation != "true":
        raise Exception('You have to set a valid value for smoothing!')
    else:
        pitch_automation = True if pitch_automation == "true" else False

    wt_automation = args.wt_automation.lower() if args.wt_automation is not None else wt_automation.lower()

    if wt_automation != "false" and wt_automation != "true":
        raise Exception('You have to set a valid value for smoothing!')
    else:
        wt_automation = True if wt_automation == "true" else False

    f0 = args.frequency
    duration = int(args.playback_length * SAMPLE_RATE)

    if 20 > f0 > 20000:
        raise Exception('Frequency has to be in the range of 20 - 20000 Hz')

    fp = glob.glob(os.path.join(args.input, '*'))

    if len(fp) == 0:
        raise Exception('No files in directory')

    stack_array = read_json_files(fp)

    osc = oscillator.WavetableOscillator(stack_array, SAMPLE_RATE, wt_automation=True)

    audio_data = osc.process(duration, f0)

    sf.write(os.path.join(args.output, f'synthesised_waveform_{os.path.basename(args.input)}.wav'), audio_data, SAMPLE_RATE)


if __name__ == '__main__':
    main()
