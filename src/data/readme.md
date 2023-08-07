## How-to: register a dataset with detectron2
We use the register_coco_instances from coco.py file. We define the classes present in annotations in list and provide it as metadata.
provide the path to the annotations and root image directory. Make sure that the path root + add_path_in_label_file provides the 
complete path to the image file.
It internally uses load_coco_json to use the COCO format json file to create dataset dictionary.

## Sampling strategies for STUD
