import numpy as np


class WavetableOscillator:
    def __init__(self, wavetable_stacks, sample_rate, wt_automation=False, pitch_automation=False):
        self.phase = 0.0
        self.increment = 0.0
        self.wt_pos = 0.0
        self.wt_stack_pos = 0
        self.wavetable_stacks = wavetable_stacks
        self.stack_length = len(wavetable_stacks[0].tables_stack)
        self.wt_automation = wt_automation
        self.pitch_automation = pitch_automation
        self.sample_rate = sample_rate

    def process(self, block_len, frequency):
        block_length = block_len

        output = np.empty(block_length)

        if self.pitch_automation:
            freq_val = frequency / self.sample_rate
            freq_mult = 1.0 + (np.log(20000.0 / self.sample_rate) - np.log(freq_val)) / block_len

        self.__set_frequency(frequency / self.sample_rate)
        start_wt = self.wt_pos
        wt_add = (1 / block_length) * (len(self.wavetable_stacks) - 1 - start_wt)
        for i in range(block_length):

            if self.pitch_automation:
                self.__set_frequency(freq_val)
                freq_val *= freq_mult

            if self.wt_automation:
                self.wt_pos += wt_add

            current_wt = self.wavetable_stacks[int(self.wt_pos)].tables_stack[self.wt_stack_pos].table
            next_wt = (self.wavetable_stacks[int(self.wt_pos + 1)].tables_stack[self.wt_stack_pos].table if
                       int(self.wt_pos + 1) < len(self.wavetable_stacks) else
                       np.zeros(512))

            # interpolate current wt
            frac_part = self.wt_pos - int(self.wt_pos)

            a = self.__linear_interpolation(current_wt, self.phase * len(current_wt))
            b = self.__linear_interpolation(next_wt, self.phase * len(next_wt))

            output[i] = a + (b - a) * frac_part

            self.__update_phase()

        return output

    @staticmethod
    def __linear_interpolation(table, phase):
        int_part = int(phase)
        frac_part = phase - int_part

        a = table[int_part]
        b = table[(int_part + 1) % len(table)]

        return a + (b - a) * frac_part

    def __update_phase(self):
        self.phase += self.increment

        if self.phase >= 1.0:
            self.phase -= 1.0

    def __set_frequency(self, normalized_freq):
        self.increment = normalized_freq
        current_stack_pos = 0

        while ((self.increment >=
                self.wavetable_stacks[int(self.wt_pos)].tables_stack[current_stack_pos].top_frequency) and
               (current_stack_pos < (self.stack_length - 1))):
            current_stack_pos += 1

        self.wt_stack_pos = current_stack_pos
