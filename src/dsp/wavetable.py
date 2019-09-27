import numpy as np
import json


class Wavetable:
    def __init__(self, top_frequency, table_length, table):
        self.top_frequency = top_frequency
        self.table_length = table_length
        self.table = np.array(table)

    @staticmethod
    def from_json_dict(dct):
        f0 = None
        l = None
        table = None

        if 'top_frequency' in dct:
            f0 = dct['top_frequency']

        if 'table_length' in dct:
            l = dct['table_length']

        if 'table' in dct:
            table = dct['table']

        if f0 is not None and l is not None and table is not None:
            return Wavetable(f0, l, table)

        return None

    def to_json_dict(self):
        return WavetableEncoder().encode(self)


class WavetableStack:
    def __init__(self, tables):
        self.tables_stack = tables

    @classmethod
    def from_json_dict(cls, dct):
        tables = list(map(Wavetable.from_json_dict, dct['tables_stack']))
        return cls(tables)

    def to_json_dict(self):
        return WavetableStackEncoder().encode(self)

    def serialize_to_file(self, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(self.to_json_dict(), f,ensure_ascii=False)


class WavetableEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, Wavetable):
            return dict(top_frequency=object.top_frequency,
                        table_length=object.table_length,
                        table=object.table.tolist())
        else:
            return super().default(object)


class WavetableStackEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, WavetableStack):
            stack_dct = []
            dct = dict(tables_stack=stack_dct)

            for wt in object.tables_stack:
                json_wt = json.loads(WavetableEncoder().encode(wt))
                dct['tables_stack'].append(json_wt)

            return dct
        else:
            return super().default(object)


def main():
    f0 = 0.5
    table = [2, 0.4, 5.]
    l = len(table)

    f0_1 = 0.2
    table_2 = [3, 4., 6.]
    l2 = len(table_2)

    wt = Wavetable(f0, l, table)
    wt2 = Wavetable(f0_1, l2, table_2)
    wt_stack = [wt, wt2]
    stack = WavetableStack(wt_stack)
    stack.serialize_to_file('wt_stack.json')

    with open('wt_stack.json') as data_file:
        data_loaded = json.load(data_file)
        data_loaded = json.loads(data_loaded)

        json_stack = WavetableStack.from_json_dict(data_loaded)
        print(json_stack)


if __name__ == '__main__':
    main()