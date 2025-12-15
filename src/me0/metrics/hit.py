from typing import Any, Tuple
from torch import Tensor
from torchmetrics.classification import BinaryRecall, BinaryStatScores
from torchmetrics.functional.classification.stat_scores import (
        _binary_stat_scores_tensor_validation,
        _binary_stat_scores_format,
        _binary_stat_scores_update,
)
from torchmetrics.functional.classification.precision_recall import _precision_recall_reduce
from torchmetrics.utilities.compute import _safe_divide
from tensordict import TensorDict

class HitEfficiency(BinaryRecall):

    def update(self, input: TensorDict) -> None:
        if self.ignore_index is not None:
            if 'data_mask' in input:
                target = input['target'].masked_fill(~input['data_mask'], self.ignore_index)
            elif 'pad_mask' in input:
                target = input['target'].masked_fill(input['pad_mask'], self.ignore_index)
            else:
                target = input['target']
        else:
            target = input['target']

        super().update(input['preds'], target)

# adapted from https://github.com/Lightning-AI/torchmetrics/blob/v1.3.2/src/torchmetrics/classification/precision_recall.py#L472-L592
class HitFakeRate(BinaryStatScores):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    def update(self, input: TensorDict) -> None:
        if self.ignore_index is not None:
            if 'data_mask' in input:
                target = input['target'].masked_fill(~input['data_mask'], self.ignore_index)
            elif 'pad_mask' in input:
                target = input['target'].masked_fill(input['pad_mask'], self.ignore_index)
            else:
                target = input['target']
        else:
            target = input['target']

        super().update(input['preds'], target)

    def compute(self) -> Tensor:
        """
        fake rate = FP / PP = 1 - TP / PP = 1 - precision
        """
        tp, fp, tn, fn = self._final_state()
        precision = _precision_recall_reduce(
            "precision", tp, fp, tn, fn, average="binary",
            multidim_average=self.multidim_average # type: ignore
        )
        return 1 - precision
