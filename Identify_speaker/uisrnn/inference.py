# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

import uisrnn

class Diarization_model():
    def __init__(self):
        model_args, training_args, self.inference_args = uisrnn.parse_arguments()
        model_args.enable_cuda = False
        model_args.rnn_depth = 1
        model_args.rnn_hidden_size = 512
        model_args.observation_dim = 256
        self.model = uisrnn.UISRNN(model_args)
        self.model.load('./saved_model.uisrnn')


    def verify_speaker(self, list_emb:list):
        return self.model.predict(np.array(list_emb), self.inference_args)




Diarization_model()
