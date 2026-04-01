# model settings
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='DecoupleNet',
        embed_dim=48,
        depths=(1, 6, 6, 2),
        att_kernel=(9, 9, 9, 9),
        mlp_ratio=2.,
        drop_path_rate=0.15,
    ),
    decode_head=dict(
        type="UNetFormerDecoder",
        in_channels=[48, 96, 192, 384],
        in_index=[0, 1, 2, 3],
        decode_channels=64,
        dropout=0.1,
        window_size=8,
        num_classes=16,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
