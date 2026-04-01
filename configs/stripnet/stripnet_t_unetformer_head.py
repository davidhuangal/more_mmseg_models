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
        type='StripNet',
        embed_dims=[32, 64, 160, 256],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        drop_rate=0.1,
        drop_path_rate=0.15,
        depths=[3, 3, 5, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    decode_head=dict(
        type="UNetFormerDecoder",
        in_channels=[32, 64, 160, 256],
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
