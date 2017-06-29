from data_generators.base_data_generator import BaseBatchGenerator


class VctkBatchGenerator(BaseBatchGenerator):
    def __init__(self, dir_path, num_speakers=100, frames=64, silence=True, val_frac=0.2, id='VCTK-Corpus'):
        super(VctkBatchGenerator, self).__init__(dir_path, num_speakers, frames, silence, val_frac, id)
