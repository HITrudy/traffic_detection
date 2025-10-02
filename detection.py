from ultralytics import YOLO
import cv2


class dectection:
    def __init__(self, ts_path, pbtl_path):
        """
        初始化两个YOLO模型
        :param model1_path: traffic sign权重文件路径
        :param model2_path: person bicycle traffic light权重文件路径
        """
        self.ts = YOLO(ts_path)     #traffic sign
        self.pbtl = YOLO(pbtl_path)   #person bicycle traffic light

    def detect_image(self, img_path):
        """
        使用指定模型检测图像
        :param img_path: 图像路径
        :return: 检测结果对象
        """
        
        res_ts = self.ts(img_path)
        res_pbtl = self.pbtl(img_path)
        return res_ts, res_pbtl
    
    def draw_and_save(self, img_path, save_path):
        """
        在图像上绘制检测框并保存
        :param img_path: 原始图像路径
        :param results: detect_image返回的结果对象
        :param save_path: 保存图像的路径
        """
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        res_ts, res_pbtl = self.detect_image(img_path)
        
        for result in res_ts:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])               #confidence 
                label = f"{result.names[cls]} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw label
                # cv2.putText(img, label, (x1, y1 - 10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for result in res_pbtl:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if(cls > 12): continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])               #confidence 
                label = f"{result.names[cls]} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw label
                # cv2.putText(img, label, (x1, y1 - 10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(save_path, img)
        print(f"已保存带检测框的图像到: {save_path}")


if __name__ == "__main__":
    p1 = "traffic_sign.pt"
    p2 = "person_car_tl.pt"

    yolod = dectection(p1,p2)
    yolod.draw_and_save("demo.png", "output.png")