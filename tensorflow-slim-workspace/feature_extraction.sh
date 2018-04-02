python example_feat_extract.py \
--network inception_resnet_v1 \
--checkpoint /data/zeng/x_model/model.ckpt-28000 \
--image_path ./images_dir/ \
--out_file ./features.h5 \
--num_classes 1000 \
--layer_names inception_resnet_v1/logits
