"""DetailDice-GC's update scope and two-head losses for an external GCNet-S."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


DETAIL_BLOCK = "backbone.detail_branch_layers.2.1"
HEAD_LR = 0.0005
BACKBONE_LR = 0.0001
DICE_WEIGHT = 0.5


def configure_detaildice(model: nn.Module) -> list[dict[str, float | list[nn.Parameter]]]:
    """Set training modes and return the SGD groups for the block and both heads."""
    backbone = model.get_submodule("backbone")
    heads = model.get_submodule("decode_head")
    detail = model.get_submodule(DETAIL_BLOCK)

    model.train().requires_grad_(True)
    backbone.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    detail.requires_grad_(True)

    return [
        {"params": list(heads.parameters()), "lr": HEAD_LR},
        {"params": list(detail.parameters()), "lr": BACKBONE_LR},
    ]


class TwoHeadCE(nn.Module):
    """Cross-entropy control on input-resolution auxiliary and main logits."""

    def forward(self, logits: tuple[Tensor, Tensor], target: Tensor) -> dict[str, Tensor]:
        aux, main = (F.cross_entropy(output.float(), target, ignore_index=255) for output in logits)
        return {"main": main, "aux": aux, "total": main + 0.4 * aux}


class TwoHeadCEDice(TwoHeadCE):
    """Add main-head soft Dice over ground-truth-present classes."""

    def forward(self, logits: tuple[Tensor, Tensor], target: Tensor) -> dict[str, Tensor]:
        losses = super().forward(logits, target)
        valid = target != 255
        scores = logits[1].movedim(1, -1)[valid].float().softmax(dim=-1)
        labels = F.one_hot(target[valid], num_classes=scores.shape[-1]).to(dtype=scores.dtype)
        gt_count = labels.sum(dim=0)
        present = gt_count > 0
        overlap = (scores * labels).sum(dim=0)
        dice_by_class = (2 * overlap + 1) / (scores.sum(dim=0) + gt_count + 1)
        dice = ((1 - dice_by_class) * present).sum() / present.sum().clamp_min(1)
        return {**losses, "main_dice": dice, "total": losses["total"] + DICE_WEIGHT * dice}
