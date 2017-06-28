from data_generators.base_data_generator import BaseBatchGenerator


class TimitBatchGenerator(BaseBatchGenerator):
    def __init__(self, dir_path, num_speakers=100, frames=64, silence=True, batch_size=32, file_batch_size=10):
        super(TimitBatchGenerator, self).__init__(dir_path, num_speakers, frames, silence, batch_size, file_batch_size)
