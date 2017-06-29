from data_generators.base_data_generator import BaseBatchGenerator


class TimitBatchGenerator(BaseBatchGenerator):
    def __init__(self, dir_path, num_speakers=100, frames=64, silence=True, file_batch_size=1, val_frac=0.2):
        super(TimitBatchGenerator, self).__init__(dir_path, num_speakers, frames, silence, file_batch_size, val_frac)
