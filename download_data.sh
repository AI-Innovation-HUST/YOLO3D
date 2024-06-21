echo "Download image"


wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip


mkdir kitti

unzip -d kitti data_object_image_2.zip


echo "Dowload labels"

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip


unzip -d kitti data_object_label_2.zip
