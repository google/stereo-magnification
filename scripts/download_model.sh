#!/bin/bash -eu
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download the SIGGRAPH 2018 model.
mkdir -p models
wget -P models https://storage.googleapis.com/stereo-magnification-public-files/models/siggraph_model_20180701.tar.gz
tar xzvf models/siggraph_model_20180701.tar.gz -C models
