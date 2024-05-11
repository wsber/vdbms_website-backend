import torch
import torchvision
from PIL import Image
from tqdm import tqdm
# from torch.utils.data import DataLoader
from udfs.yolov5.utils.general import non_max_suppression, scale_coords



class YoloWrapper:

    def __init__(self, model_name = 'yolov5s', device=None):
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        assert(model_name in ['yolov5s', 'yolov5m6', 'yolov5s6', 'yolov5n6', 'yolov5m'])
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)


    def inference(self, images, new_shape = None, old_shape = None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device,288)
        dataset = InferenceDataset(images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1)
        self.model.to(device)
        self.model.eval()
        conf = 0.01  # NMS confidence threshold
        iou = 0.45
        classes = None
        organized_output = []
        index_cache = {}
        iid = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # print('[len_batch]: ',len(batch))
                batch = batch.to(device)
                output = self.model(batch)
                tt = output.clone()
                y = non_max_suppression(tt, conf_thres=conf, iou_thres=iou, classes=classes)  # NMS
                # if new_shape is not None and old_shape is not None:
                #     for i in range(len(y)):
                #         scale_coords(new_shape, y[i], old_shape)
                # print(y,'y is here')
                ### now y is a list so we must do it one by one
                for prediction in y:
                    out = {}
                    prediction = prediction.to('cpu')
                    boxes = prediction[:, :4]
                    scores = prediction[:, 4]
                    labels = prediction[:, 5].int()
                    out['boxes'] = boxes
                    out['labels'] = labels
                    out['scores'] = scores
                    for box, score, label in zip(boxes, scores, labels):
                        if self.model.names[int(label)] == 'car':
                            inner_dict = {
                                'object_name': self.model.names[int(label)],
                                'confidence': score,  # Convert tensor to float
                                'xmin': int(box[0]),
                                'ymin': int(box[1]),
                                'xmax': int(box[2]),
                                'ymax': int(box[3])
                            }
                            if iid in index_cache :
                                index_cache[iid].append(inner_dict)
                            else:
                                index_cache[iid] = [inner_dict]
                    # print(out)
                    # organized_output.append(out)

        classes = self.model.names

        organized_dict = {
            'categories' : classes,
            'annotations': index_cache
        }
        # print(organized_dict)
        # print(len(organized_output))
        return organized_dict

    def inference_o(self, images, new_shape = None, old_shape = None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        print(device,388)
        # Split the list of images into batches of 8
        organized_output = []
        self.model.to(device)
        self.model.eval()
        pil_images = [Image.fromarray(img) for img in images]
        image_batches = [pil_images[i:i + 8] for i in range(0, len(images), 8)]
        for batch in tqdm(image_batches,desc='iterate over images by yolov5s'):
            results = self.model(batch)
            # print('[Index_result]: ', results)
            for result in results.xyxy:
                # print('[Index_result]: ', result)
                detections = result
                if len(detections) > 0:
                    # Iterate through detections for this image
                    for detection in detections:
                        # Extract bounding box coordinates and confidence score
                        bbox = detection[:4]  # First four elements are usually bbox coordinates
                        confidence = detection[4]  # Fifth element is often the confidence score
                        object_name = self.model.names[int(detection[5])]  # Get class name using model.names
                        if object_name == 'car' :
                            # Create the inner dictionary
                            inner_dict = {
                                'object_name': object_name,
                                'confidence': confidence.item(),  # Convert tensor to float
                                'xmin': int(bbox[0]),
                                'ymin': int(bbox[1]),
                                'xmax': int(bbox[2]),
                                'ymax': int(bbox[3])
                            }
                            organized_output.append(inner_dict)
        classes = self.model.names
        organized_dict = {
            'categories': classes,
            'annotations': organized_output
        }
        # print(organized_dict)
        # print(len(organized_output))
        return organized_dict

    # 定义预处理函数，根据你的模型需求进行修改
    def preprocess_image(self, img):
        # 例如：将图片转换为tensor并进行归一化
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),  # 将图片调整为 224x224
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img)

def inference_transforms():
    ttransforms = torchvision.transforms.Compose([
        # transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((320, 320)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return ttransforms



class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.transform = inference_transforms()
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
