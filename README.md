# Traffic-Signs-Recognition-Detection
Traffic Sign detection and recognition using the Tensorflow 

Part2(Android):[Android](https://github.com/chronis98/Traffic-Sign-Recognition-Detection-Android)
# 1.Data
On this project data was extracted from German Traffic Sign Dataset.More precisely the [GTSDB](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset), was used for traffic sign recognition on images with background environment and various light conditions and the [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=datasetGTSB, was used on training traffic sign classes from cropped road sign images.<br />
GTSB: 222 manually labeled images for test ,440 manually labeled images for training<br />
GTSRB: 12630 annotated images for test, more than 50k images in total for training
# 2.Implementation
This project is using transfer learning method, through retraining pretrained models from google.
After trying multiple variations of pretrained models(MobilenetV1,MobilenetV2,InceptionV2 etc..) and datasets, i came to the conclusion that the most efficient in accuracy/speed of detection in live feeds, that would be deployed on android after conversion to Tensorflow Lite was two custom MobilenetV1 models.The first model was trained on GTSB dataset for detecting if there is a traffic sign in the background and the second model was trained on GTSRB to identify the class of the identified model.The first model was trained for about 4 hours and second models for about 8 hours in order to drop loss consistently  below 1.99.<br />
Hardware specs: i5-6600k @3.5ghz, 16GB ddr4 ram @2133mhz,Geforce Gtx 1060 6gb 
## -TfRecord generate
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record<br />
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record<br />

For the first model:<br />
## -Training configuration
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
## -Frozen graph export
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config trained_checkpoint_prefix training/model.ckpt-14131 --output_directory inference_graph<br /><br />

Same technique was used for the second model.

# 3.Results
After creating custom python scripts for image, video detection that make use of the two models, i was able to get some decent results, as seen on the images below
## -Class Metadata
<img src="https://github.com/chronis98/Traffic-Signs-Recognition-Detection/blob/master/meta.jpg" width="350"> 

## -Image Detection/Recognition
<img src="https://github.com/chronis98/Traffic-Signs-Recognition-Detection/blob/master/Screenshot_1.png" width="550" > 
<img src="https://github.com/chronis98/Traffic-Signs-Recognition-Detection/blob/master/Screenshot_3.png" width="550" > 
<img src="https://github.com/chronis98/Traffic-Signs-Recognition-Detection/blob/master/Screenshot_5.png" width="550"> 
<img src="https://github.com/chronis98/Traffic-Signs-Recognition-Detection/blob/master/Screenshot_6.png" width="550" > 
<img src="https://github.com/chronis98/Traffic-Signs-Recognition-Detection/blob/master/Screenshot_7.png" width="550" > 

