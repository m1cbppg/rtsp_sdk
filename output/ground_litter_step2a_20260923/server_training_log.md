# Step 2A CUDA tiny overfit - condensed training log

Source: /home/sf01/step2a-20260923/logs/train_batch8.log (ANSI progress stripped, duplicate per-epoch lines collapsed).

```
YOLO26s summary: 260 layers, 9,948,638 parameters, 9,948,638 gradients, 22.7 GFLOPs
Transferred 696/708 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias(decay=0.0)
Logging results to /home/sf01/step2a-20260923/out/runs/tiny_overfit
Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
all         22         41       0.05     0.0488      0.012    0.00762
all         22         41     0.0289     0.0732    0.00973    0.00613
all         22         41     0.0141       0.22    0.00653     0.0037
all         22         41    0.00715      0.293    0.00543     0.0028
all         22         41    0.00504      0.439    0.00451    0.00217
all         22         41    0.00395      0.537    0.00416    0.00204
all         22         41    0.00369      0.561    0.00414    0.00213
all         22         41    0.00384      0.585     0.0042    0.00225
all         22         41     0.0323      0.463     0.0317    0.00589
all         22         41     0.0422      0.268     0.0281    0.00532
all         22         41     0.0591      0.244     0.0359    0.00632
all         22         41     0.0648       0.22      0.035    0.00651
all         22         41    0.00577     0.0488   0.000772   0.000235
all         22         41    0.00167      0.268   0.000569   0.000198
all         22         41    0.00335      0.537     0.0127    0.00318
all         22         41      0.131      0.146      0.065      0.031
all         22         41     0.0798       0.22     0.0456     0.0225
all         22         41     0.0538      0.122     0.0645     0.0329
all         22         41      0.123      0.146     0.0687     0.0329
all         22         41       0.18       0.39       0.24      0.142
all         22         41      0.532       0.61      0.599       0.35
all         22         41      0.711      0.634      0.664      0.421
all         22         41       0.84      0.771      0.803      0.583
all         22         41      0.872      0.683      0.841      0.642
all         22         41      0.929      0.641      0.824      0.613
all         22         41      0.936      0.708      0.849      0.661
all         22         41      0.895      0.805      0.915      0.723
all         22         41      0.875      0.854      0.913      0.768
all         22         41      0.973      0.868      0.946       0.78
all         22         41      0.949        0.9      0.961      0.791
all         22         41       0.95      0.925       0.97      0.834
all         22         41       0.95      0.951      0.974      0.859
all         22         41      0.951       0.95      0.976      0.871
all         22         41      0.929       0.96       0.98      0.873
all         22         41      0.929      0.976      0.983      0.887
all         22         41      0.975      0.946      0.987      0.882
all         22         41      0.975      0.948      0.989        0.9
all         22         41      0.975      0.976       0.99      0.911
all         22         41      0.976      0.975      0.991      0.925
all         22         41      0.949      0.976      0.991      0.936
all         22         41      0.976      0.975      0.993      0.935
all         22         41      0.953      0.993      0.992      0.929
all         22         41      0.976      0.994      0.993      0.946
all         22         41      0.953      0.995      0.992      0.949
100 epochs completed in 0.046 hours.
Validating /home/sf01/step2a-20260923/out/runs/tiny_overfit/weights/best.pt...
YOLO26s summary (fused): 120 layers, 9,465,567 parameters, 0 gradients, 20.8 GFLOPs
all         22         41      0.976      0.995      0.993       0.95
Speed: 0.4ms preprocess, 24.3ms inference, 0.0ms loss, 1.5ms postprocess per image
"initial_box_loss": 1.61586,
"final_box_loss": 0.28691,
```
