# Copyright 2025 Huawei Technologies Co., Ltd
#
# Adapted from https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/ocr.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from typing import Union

import numpy as np
import torch
from Levenshtein import distance
from PIL import Image

from .scorer import Scorer


class PaddleOCRScorer(Scorer):
    def __init__(self):
        """
        OCR reward calculator
        """
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            ocr_version="PP-OCRv4",
            device="cpu",
            enable_mkldnn=False,
        )

    @torch.no_grad()
    async def __call__(
        self,
        images: Union[list[Image.Image], np.ndarray, torch.Tensor],
        prompts: list[str],
    ):
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward scores
        """
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images = self.array_to_images(images)

        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), (
            "Images and prompts must have the same length"
        )

        gts = []
        for prompt in prompts:
            try:
                # in case the data_source is `prompt`, extract the text between quotes
                text = prompt.split('"')[1]
            except IndexError:
                # in case the data_source is `ocr`, since the text is already extracted from dataloader, use it directly
                text = prompt
            gts.append(text)

        for img, gt in zip(images, gts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition
                result = self.ocr.predict(img)
                recognized_text = ""
                if result[0]:
                    rec_texts = result[0]["rec_texts"]
                    rec_scores = result[0]["rec_scores"]
                    # Extract recognized text (handle possible multi-line results)
                    recognized_text = "".join(
                        [
                            rec_texts[idx] if score > 0 else ""
                            for idx, score in enumerate(rec_scores)
                        ]
                    )

                recognized_text = recognized_text.replace(" ", "").lower()
                gt = gt.replace(" ", "").lower()
                if gt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, gt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(gt):
                    dist = len(gt)

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(gt)  # Maximum penalty
            reward = 1 - dist / (len(gt))
            rewards.append(reward)

        return rewards


async def compute_score(images, prompts):
    """
    Compute OCR reward score using PaddleOCR for a batch of images and prompts.
    """
    scorer = PaddleOCRScorer()
    scores = await scorer(images, prompts)

    return scores
