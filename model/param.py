class Param:
    def __init__(self, **kwargs):
        self.max_normal_deviation = 0.20
        self.max_num_elements_single_cluster = 12
        self.ps_upper_bound = 0.90

        self.__dict__.update(kwargs)
