SHOP
peopleCounter-SSD.py -r 320 -zx1 10 -zy1 100 -zx2 1000 -zy2 600 --input Files/shop.mp4 --prototxt Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt --model Files/frozen_inference_graph.pb

ENTRANCE1
peopleCounter-SSD.py -r 320 -zx1 100 -zy1 300 -zx2 800 -zy2 720 --input Files/door_entrance1.mp4 --prototxt Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt --model Files/frozen_inference_graph.pb


ENTRANCE2
peopleCounter-SSD.py -r 320 -zx1 100 -zy1 200 -zx2 300 -zy2 600 --input Files/door_entrance2.mp4 --prototxt Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt --model Files/frozen_inference_graph.pb