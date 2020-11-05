from .test_options import TestOptions

class ConvertOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--model_in', type=str, default='')
        self.parser.add_argument('--model_out', type = str, default = '')