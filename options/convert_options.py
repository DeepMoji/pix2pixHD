from .test_options import TestOptions

class ConvertOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--model_in', type=str, default='')
        self.parser.add_argument('--model_out', type = str, default = '')
        self.parser.add_argument('--Convert32to16', action='store_true',
                                 help='if specified, the method convert f32 mlmodel')