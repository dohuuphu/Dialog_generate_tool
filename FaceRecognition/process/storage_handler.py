from builtins import object
import cv2
import os
import json

import utils
from config.config import config


def get_data_kvqa_from_storage(storage_path):
    result = []

    origin_path, graph_info_path, _, _ = config["storage"]["kvqa"]["sub_path"]
    list_img_name = sorted([file for file in os.listdir(os.path.join(storage_path, origin_path))], reverse=True)
    for idx, img_name in enumerate(list_img_name):
        f_graph_info = open(os.path.join(storage_path, graph_info_path, "{}.json".format(os.path.splitext(img_name)[0])), "r")
        graph_info = json.load(f_graph_info)
        result.append({"id": str(idx), "image_id": os.path.splitext(img_name)[0], "graph_info": graph_info})

    return result


def get_image_from_storage(storage_path, image_id):
    origin_path, graph_info_path, _, _ = config["storage"]["kvqa"]["sub_path"]
    origin_img = cv2.imread(os.path.join(storage_path, origin_path, "{}.jpg".format(image_id)))

    f_graph_info = open(os.path.join(storage_path, graph_info_path, "{}.json".format(image_id)), "r")
    graph_info = json.load(f_graph_info)

    list_objects = graph_info["objects"]
    list_relations = graph_info["relations"]

    rects = [obj["rect"] for obj in list_objects]
    labels = [obj["class"] for obj in list_objects]
    confidences =  [obj["conf"] for obj in list_objects]
    image_visualize = utils.draw_boxes_info(origin_img, rects, labels, confidences)

    graph_visualize_image = utils.draw_scene_graph_relation(list_objects, list_relations)

    origin_img = utils.get_bytes_value(cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR))
    image_visualize = utils.get_bytes_value(cv2.cvtColor(image_visualize, cv2.COLOR_RGB2BGR))
    graph_visualize = utils.get_bytes_value(cv2.cvtColor(graph_visualize_image, cv2.COLOR_RGB2BGR))
    
    result = {"image_id": image_id, 
            "origin_image": origin_img, 
            "image_visualize": image_visualize,
            "graph_visualize": graph_visualize}

    return result


def get_subject_image_from_storage(storage_path, image_id, object_id):
    origin_path, graph_info_path, _, _ = config["storage"]["kvqa"]["sub_path"]
    origin_img = cv2.imread(os.path.join(storage_path, origin_path, "{}.jpg".format(image_id)))

    f_graph_info = open(os.path.join(storage_path, graph_info_path, "{}.json".format(image_id)), "r")
    graph_info = json.load(f_graph_info)
    objects = graph_info["objects"]
    attributes = graph_info["attributes"]
    
    obj_idx = utils.find_index_in_list_dict(objects, ["id"], [object_id])
    attr_idx = utils.find_index_in_list_dict(attributes, ["id"], [object_id])
    
    obj_info = objects[obj_idx]
    rect = obj_info["rect"]
    label = obj_info["class"]
    confidence =  obj_info["conf"]
    
    image_visualize = utils.draw_boxes_info(origin_img, [rect], [label], [confidence])
    image_visualize = utils.get_bytes_value(cv2.cvtColor(image_visualize, cv2.COLOR_RGB2BGR))
    result = {
        "object_name": label,
        "image_visualize": image_visualize,
        "attribute": attributes[attr_idx]["attribute"]
    }

    return result
