import json
import random

# Load the COCO JSON file
with open('/home/mila/v/vaibhav.jade/scratch/intern/stud/datasets/coco/annotations/instances_val2017.json', 'r') as f:
    coco_data = json.load(f)
things_classes = [
                        "bicycle","bus","car","lane","lanes","motorcycle","person",
                        "roadwork_tcd","speed_limit","stop sign", "traffic light",
                        "truck"
                          ]
###  id class -1 means not things class(esmart) not found in COCO
id_classes = [2,6,3,-1,-1,4,1,-1,-1,13,10,8]
ood_classes = [i for i in range(1,91) if i not in id_classes]

new_annotations = []
for annotation in coco_data['annotations']:
    if str(annotation['image_id']).startswith('1'):
        if annotation['category_id'] in id_classes:
            new_cat_id = 1
        else:
            new_cat_id = 2
        new_annotations.append({
            'area':annotation['area'],
            'iscrowd':annotation['iscrowd'],
            'image_id':annotation['image_id'],
            'bbox':annotation['bbox'],
            'category_id':new_cat_id,
            'id':annotation['id']
        })
ood_categories = [{"supercategory": "ID","id": 1,"name": "ID"},{"supercategory": "OOD","id": 2,"name": "OOD"}]
ood_coco_data = {
    'images': coco_data['images'],
    'annotations': new_annotations,
    'categories': ood_categories
}
# Change the category_id for the first half to 1 and the second half to 2
# for i, annotation in enumerate(coco_data['annotations']):
#     if i < half_index:
#         annotation['category_id'] = 1
#     else:
#         annotation['category_id'] = 2

# Save the updated JSON to a new file
with open('/home/mila/v/vaibhav.jade/scratch/intern/stud/datasets/coco/instances_val2017_ood_wrt_esmart.json', 'w') as f:
    json.dump(ood_coco_data, f)

# print("Updated category IDs in annotations and saved to 'path_to_your_modified_coco_val.json'")
