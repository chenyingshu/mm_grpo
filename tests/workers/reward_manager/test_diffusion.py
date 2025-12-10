# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pytest
import numpy as np
import torch

from gerl.protocol import DataProto
from gerl.workers.reward_manager.diffusion import DiffusionRewardManager, DiffusionBatchRewardManager

@pytest.fixture
def mock_data() -> DataProto:
    test_prompt = "a photo of a cat"
    response = np.random.randn(1, 3, 64, 64)  # Example image tensor
    data = DataProto(
        batch={"responses": response},
        non_tensor_batch={
            "prompt": np.array([test_prompt]),
            "reward_model": {
                "ground_truth": test_prompt,
            },
            "data_source": "ocr", # Use CPU reward for testing
        },
    )
    return data


class TestDiffusionRewardManager():
    def setup_class(self):
        self.reward_manager = DiffusionRewardManager(None, 1, None, )

    def test_call(self, mock_data: DataProto):
        # test return dict result
        dict_result = self.reward_manager(mock_data, return_dict=True)
        expected_dict_keys = ["reward_tensor", "reward_extra_info"]
        for key in expected_dict_keys:
            assert key in dict_result, f"Key {key} not found in dict result."

        # test return tensor result
        tensor_result = self.reward_manager(mock_data, return_dict=False)
        assert isinstance(tensor_result, torch.tensor), "Tensor result is not a torch.Tensor."
        assert tensor_result.shape == (len(mock_data.batch["response"]), ), (
            f"Expected tensor shape {(len(mock_data.batch['response'],),)}, got {tensor_result.shape}."
        )

        # test NotImplementedError for unsupported rm_scores
        mock_data.batch["rm_scores"] = torch.tensor([1.0])
        with pytest.raises(NotImplementedError):
            self.reward_manager(mock_data)

class TestDiffusionBatchRewardManager():
    def setup_class(self):
        self.reward_manager = DiffusionBatchRewardManager(None, 1, None, )

    def test_call(self, mock_data: DataProto):
        # test return dict result
        dict_result = self.reward_manager(mock_data, return_dict=True)
        expected_dict_keys = ["reward_tensor", "reward_extra_info"]
        for key in expected_dict_keys:
            assert key in dict_result, f"Key {key} not found in dict result."

        # test return tensor result
        tensor_result = self.reward_manager(mock_data, return_dict=False)
        assert isinstance(tensor_result, torch.tensor), "Tensor result is not a torch.Tensor."
        assert tensor_result.shape == (len(mock_data.batch["response"]), ), (
            f"Expected tensor shape {(len(mock_data.batch['response'],),)}, got {tensor_result.shape}."
        )

        # test NotImplementedError for unsupported rm_scores
        mock_data.batch["rm_scores"] = torch.tensor([1.0])
        with pytest.raises(NotImplementedError):
            self.reward_manager(mock_data)
