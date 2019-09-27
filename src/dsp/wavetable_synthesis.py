import numpy as np
import wavetable
import scipy.fftpack


class WavetableSynthesis:
    def __init__(self, table, sample_rate=44100, spacing_st=3):
        self.tables = table
        self.sample_rate = sample_rate
        self.spacing = spacing_st

    # first calculate number of harmonics used for lowest lookup table
    # start freq is the starting frequency for the wavetable
    # sample rate is the used playback samplerate
    # spacing is the table size in semitones

    def calculate_num_harmonics(self, start_freq, sample_rate, spacing):
        return int((sample_rate / 2) / (2 ** (spacing / 12) * start_freq))

    # calculate the number of tables needed for a specified frequency range
    # start_freq is startpoint of the intervall
    # end_freq is the endpoint of the intervall
    # spacing is the table size in semitones

    def calculate_num_tables(self, start_freq, end_freq, spacing):
        tables_per_octave = 12 / spacing

        num_octaves = np.log2(end_freq / start_freq)

        return round(tables_per_octave * int(num_octaves)) + 1

    # Rounds input number up to nearest power of 2

    def round_to_nearest_power_2(self, number):
        v = np.uint32(number)

        v -= 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v += 1

        return v

    # Calculates the needed table_length
    # num_harmonics the number of harmonics returned by calculate_num_harmonics()
    # oversampling: the oversampling rate needs to be a power of two

    def calculate_table_len(self, num_harmonics, oversampling):
        table_len = 2 * num_harmonics + 1

        table_len *= oversampling

        return int(self.round_to_nearest_power_2(table_len))

    @staticmethod
    def normalise(x):
        maximum = max(abs(x))

        if maximum == 0:
            return np.zeros(len(x))

        mult = 1.0 / maximum * .999

        return x * mult

    @staticmethod
    def stretch_table(x, out_len):
        len_x = len(x)
        inc = len_x / out_len
        phase = 0

        y = np.empty(out_len)

        for i in range(out_len):
            int_part = int(phase)
            frac_part = phase - int_part

            a = x[int_part % len_x]
            b = x[(int_part + 1) % len_x]

            y[i] = a + (b - a) * frac_part

            phase += inc

        return y

    @staticmethod
    def resynthesise_table(max_num_harms, x):
        X = scipy.fftpack.fft(x)
        harm = int(max_num_harms)
        x1 = X[harm:int(len(X) / 2)]
        x2 = X[int(len(X) / 2):len(X) - harm]
        x1.fill(0)
        x2.fill(0)

        y = scipy.fftpack.ifft(X)
        y = y.real

        maximum = max(abs(y))
        mult = maximum * .999

        y /= mult

        return y

    def create_table_stack(self, table, start_freq, end_freq, num_harms, table_len, table_factor):
        tables = []
        freq = start_freq

        while True:
            fft_table = self.resynthesise_table(num_harms, table)

            freq *= table_factor
            normalised_freq = (freq / self.sample_rate)

            tables.append(wavetable.Wavetable(normalised_freq, table_len, fft_table))

            # We are finished when we synthesized just one harmonic
            if(num_harms < 2):
                break

            num_harms *= 1 / table_factor

        return wavetable.WavetableStack(tables)

    def find_start(self, data):

        for i in range(len(data)):
            if data[i] != 0.0:
                return i

    def synthesise_all_frames(self):
        sample_rate = self.sample_rate
        spacing = self.spacing
        start_freq = 20
        end_freq = 20000
        table_factor = 2 ** (spacing / 12)
        num_harms = self.calculate_num_harmonics(start_freq, sample_rate, spacing)
        num_tables = self.calculate_num_tables(start_freq, end_freq, spacing)
        table_len = self.calculate_table_len(num_harms, 2)
        table = self.tables

        # create fft tables
        table_stack = self.create_table_stack(table, start_freq, end_freq, num_harms, table_len, table_factor)

        return table_stack
