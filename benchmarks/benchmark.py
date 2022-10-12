class BaseBenchmark:

    def __init__(self, path_to_json_file: str):

        self.path_to_json_file = path_to_json_file

    def _load_benchmark(self):

        raise NotImplementedError('Please implement the load_benchmark method')

    def load_dataset_names(self):
        raise NotImplementedError('Please implement the load_dataset_names method')

    def get_hyperparameter_candidates(self):

        raise NotImplementedError('Please extend the get_hyperparameter_candidates method')

    def get_performance(self, hp_index: int, budget: int):

        raise NotImplementedError('Please extend the get_performance method')
