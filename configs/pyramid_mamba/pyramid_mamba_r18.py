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
        type="TIMMBackbone",
        model_name="resnet18",
        features_only=True,
        pretrained=True,
        in_channels=3,
        out_indices=(1, 4),
    ),
    decode_head=dict(
        type="PyramidMambaHead",
        in_channels=[64, 512],
        in_index=[0, 1],
        decoder_channels=128,
        last_feat_size=16,
        num_classes=16,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
