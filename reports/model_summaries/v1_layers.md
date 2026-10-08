| name        | type                    | output_shape   |   parameter_count | description                                                                          |
|:------------|:------------------------|:---------------|------------------:|:-------------------------------------------------------------------------------------|
| ConvBlock 1 | Sequential              | (32, 32, 32)   |               896 | Extracts basic features like edges and colors.                                       |
| ConvBlock 2 | Sequential              | (64, 16, 16)   |             18496 | Combines basic features into simpler shapes and textures.                            |
| ConvBlock 3 | Sequential              | (128, 8, 8)    |             73856 | Builds complex patterns representing traffic sign parts.                             |
| Flatten     | Flatten                 | (8192,)        |                 0 | Transforms the 3D feature maps into a 1D vector.                                     |
| Dense 1     | Linear + ReLU + Dropout | (256,)         |           2097408 | Learns high-level combinations of features, discarding some to prevent memorisation. |
| Dense 2     | Linear                  | (75,)          |             19275 | Outputs the final classification scores for each traffic sign class.                 |