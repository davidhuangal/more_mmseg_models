<div align="center">
  <img src="resources/logo.png" width="600"/>
</div>


## Introduction
This project is dedicated to implementing / adapting more models for use with the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/) framework.

## Model Zoo
<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td><b>Supported Backbones</b></td>
      <td><b>Supported Heads</b></td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="models/backbones/efficientvit.py">EfficientViT (ICCV 2023)</a></li>
          <li><a href="models/backbones/stripnet.py">StripNet (ArXiv 2025)</a></li>
          <li><a href="models/backbones/lsknet.py">LSKNet (IJCV 2024)</a></li>
          <li><a href="models/backbones/decouplenet.py">DecoupleNet (TGRS 2024)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="models/decode_heads/efficientvit_head.py">EfficientViT Head (ICCV 2023)</a></li>
          <li><a href="models/decode_heads/unetformer_head.py">UNetFormer Head (ISPRS 2022)</a></li>
          <li><a href="models/decode_heads/pyramid_mamba_head.py">PyramidMamba Head (GeoSeg)</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

Pretrained weights are available on [Google Drive](https://drive.google.com/drive/folders/131IA7V0TIt3pj4cApicav6MBo2rQFQ95?usp=sharing).
