import numpy as np
import scipy.fftpack
import YIN as pitch_detect


class AudioSlicer:
    def __init__(self, data, sample_rate, wavetable_size, smoothing, max_tables):
        self.data = data
        self.sample_rate = sample_rate
        self.yin_size = 2048
        self.detector = pitch_detect.YIN(2048, sample_rate)
        self.wavetable_size = wavetable_size
        self.smooth_length = 512
        self.smoothing = smoothing
        self.max_tables = max_tables

    @staticmethod
    def stretch_table(x, out_len):
        len_x = len(x)
        inc = len_x / out_len
        phase = 0

        y = []

        for i in range(out_len):
            int_part = int(phase)
            frac_part = phase - int_part

            a = x[int_part % len_x]
            b = x[(int_part + 1) % len_x]

            y.append(a + (b - a) * frac_part)

            phase += inc

        return y

    def pad_zero(self):
        zero_buf = np.zeros(len(self.data), dtype=float)
        self.data = np.append(self.data, zero_buf)

    @staticmethod
    def filter_dc(frame):
        frame = np.array(frame)
        avg = frame.sum() / len(frame)
        avg_arr = np.repeat(avg, len(frame))

        return frame - avg_arr

    @staticmethod
    def normalise(frame):
        maximum = np.max(np.abs(frame))

        if maximum == 0:
            return np.zeros(len(frame))

        mult = 1.0 / maximum * .999

        return np.array(frame) * mult

    @staticmethod
    def shift_frame(frame):
        frame = np.array(frame)
        maximum = np.abs(np.max(frame))
        minimum = np.abs(np.min(frame))
        dist = maximum + minimum
        dist = int(dist / 2)

        if maximum > minimum:
            frame = frame - np.repeat(dist, len(frame))
        else:
            frame = frame + np.repeat(dist, len(frame))

        return frame

    def tanh_smoothing(self, frame):
        half_fade = int(self.smooth_length / 2)
        manipulation_frame = np.append(frame[-half_fade:], frame[:half_fade])
        x = np.arange(half_fade)
        b = half_fade / 6
        s = 0.5 + 0.5 * np.tanh((x - half_fade / 2) / b)

        frame[:half_fade] = manipulation_frame[half_fade:] * s
        frame[-half_fade:] = manipulation_frame[:half_fade] * (1-s)

        return frame

    @staticmethod
    def cross_corr(x, y):
        return scipy.fftpack.ifft(scipy.fftpack.fft(x) * scipy.fftpack.fft(y).conj()).real

    def circular_shift(self, frame1, frame2):

        r = self.cross_corr(frame2, frame1)

        shift = np.argmax(r)

        half_len = len(frame1) // 2

        if shift > half_len:
            shift = -shift

        return np.roll(frame2, shift)

    def slice(self):
        start_sample = 0
        progress = self.yin_size
        frame_end = progress
        file_len = len(self.data)

        frames = []

        self.pad_zero()

        while start_sample < file_len and len(frames) <= self.max_tables:
            tau = self.detector.detect_pitch(self.data[start_sample:frame_end])
            frame = self.stretch_table(self.data[start_sample:int(start_sample + tau)], self.wavetable_size)
            frame = self.filter_dc(frame)

            start_sample += int(tau)
            frame_end = start_sample + progress

            if np.sum(np.abs(frame)) <= 10e-1 or tau <= 2.0:
                continue

            frames.append(frame)

        frames = np.array(frames).reshape(-1, self.wavetable_size)
        prev_frame = None

        for f in frames:
            if self.smoothing:
                f = self.tanh_smoothing(f)
            f = self.normalise(f)
            if prev_frame is not None:
                f = self.circular_shift(prev_frame, f)
            prev_frame = f

        return frames
