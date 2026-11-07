# DetailDice-GC

DetailDice-GC adapts a Cityscapes-pretrained GCNet-S to CamVid with 37 labeled training frames. It updates the terminal detail block and both segmentation heads and adds class-averaged soft Dice to the main-head objective. This repository contains the study's update policy, loss code, fixed subset IDs, protocol and archived aggregate scores. The backbone, data and source checkpoint come from their original providers. The example below integrates the method into an existing training loop. The historical training and held-out evaluation runners and their checkpoints remain in the study archive.

## External inputs

- Use the expanded GCNet-S N=2 architecture with the Cityscapes batch-normalization configuration from the pinned [GCNet author repository](https://github.com/gyyang23/GCNet/tree/8b4c0abf82cfe13a39ae10d8431a554fc5b3284b). The authors' setup instructions cover their MMSegmentation fork and environment. The study used a local PyTorch port of that architecture with PyTorch 2.7.1.
- Initialize from the authors' [Cityscapes GCNet-S checkpoint](https://drive.google.com/file/d/1KersBP95k3b0AELiYlQ1rk4PKUmN-ueu/view), listed at 76.9 mIoU. Its SHA-256 is in [`protocol.json`](protocol.json). Build an 11-class model, initialize it under the training seed, then copy source state tensors except the weights and biases of `decode_head.conv_seg` and `decode_head.aux_cls_seg_c4`. Those four tensors retain their 11-class initialization. Check that the source and target keys match and load the resulting state strictly.
- Get the 11-class images, masks and original validation and test splits from the pinned [SegNet CamVid distribution](https://github.com/alexgkendall/SegNet-Tutorial/tree/fcaf7c4978dd8d091ec67db7cb7fdd225f5051c5/CamVid). The study-specific training lists contain 37 PNG basenames each in [`subset-a.txt`](subset-a.txt) and [`subset-b.txt`](subset-b.txt). Use the original `val.txt` and `test.txt` paired entries from SegNet, sorted by image basename.

## Use the method in a PyTorch loop

The model must expose `backbone.detail_branch_layers.2.1` and `decode_head`. The head must produce `(auxiliary_logits, main_logits)` in training mode. The example uses the pinned MMSegmentation model's default tensor forward and resizes both logits to the input geometry. Supply normalized float RGB `images` of shape `[N, 3, H, W]` and integer `labels` of shape `[N, H, W]` on the same device as the initialized model.

```python
import torch
from torch.nn import functional as F

from detaildice_gc import TwoHeadCEDice, configure_detaildice

# model is the initialized external 11-class GCNet-S, already on the training device.
groups = configure_detaildice(model)
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 474_902
optimizer = torch.optim.SGD(groups, momentum=0.9, weight_decay=0.0005)
criterion = TwoHeadCEDice()

# One loss calculation inside the caller's loop.
targets = labels.long().masked_fill(labels == 11, 255)
aux, main = model(images)
logits = (
    F.interpolate(aux, size=images.shape[-2:], mode="bilinear", align_corners=False),
    F.interpolate(main, size=images.shape[-2:], mode="bilinear", align_corners=False),
)
loss = criterion(logits, targets)["total"]
```

Pass batches containing at least one non-void pixel. Cross-entropy averages over valid pixels and the Dice term pools valid pixels across the batch, then averages over classes present in the ground truth. `configure_detaildice` puts the model in training mode and freezes batch-normalization running statistics while allowing the affine weights in the selected modules to train. After validation, restore the original module modes or call `configure_detaildice(model)` again to restore training modes. Keep the existing optimizer and scheduler state when resuming the loop.

The study protocol records the input normalization, SGD schedule, mixed precision, evaluation procedure and matched controls. The aggregate scores in [`locked-results.json`](locked-results.json) describe the six archived subset-A checkpoints at 800 updates. Those scores document the historical comparison. The upstream integration example is based on the pinned model's interface and has been checked by source inspection.
