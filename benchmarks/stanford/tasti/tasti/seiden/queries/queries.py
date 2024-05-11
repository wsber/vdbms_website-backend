import benchmarks.stanford.tasti.tasti.query as tasti


class NightStreetAggregateQuery(tasti.AggregateQuery):
    # def score(self, target_dnn_output):
    #     return len(target_dnn_output)
    def __init__(self,index):
        super().__init__(index)
        self.call_count = 0
    def score(self, results):
        # return len(results)
        # self.call_count += 1
        # print("[call_count]: ",self.call_count)
        # print("[len_results]", len(results))
        if len(results) > 1 :
            # print("[len_results]", len(results))
            return  len(results)
        elif len(results) == 1:
            if results[0]['confidence'] == -1 :
                # print("[results]", 0)
                return 0
            else:
                # print("[results]", len(results))
                return 1
        else :
            # print("[results]", 0)
            return  0

class NightStreetAveragePositionAggregateQuery(tasti.AggregateQuery):
    def __init__(self, index, im_size = 1750):
        super().__init__(index)
        self.im_size = im_size

    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            avg = 0.
            if len(boxes) == 0:
                return 0.
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                avg += x / self.im_size
            return avg / len(boxes)
        return proc_boxes(target_dnn_output)

class NightStreetLimitQuery(tasti.LimitQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)

class NightStreetSUPGPrecisionQuery(tasti.SUPGPrecisionQuery):
    # def score(self, target_dnn_output):
    #     return 1.0 if target_dnn_output > 0 else 0.0
    def score(self, results):
        # print('[results]: ',results)
        if(len(results) ==0):
            return 0
        else:
            max_confidence = results[0]['confidence']
            return 1.0 if max_confidence > 0 else 0.0
        # max_person_confidence = 0.0
        # # 遍历检测结果中的每个预测框
        # for row in results:
        #     # 如果预测框的类别是 person
        #     if row['object_name'] == 'car':  # person 类别的索引为 0
        #         # 更新最高置信度的值
        #         max_person_confidence = max(max_person_confidence, row['confidence'])
        # 输出最高置信度的 person 类别预测框的置信度
        # print("最高置信度的 car 类别预测框的置信度:", max_person_confidence)
        # return 1.0 if max_person_confidence > 0 else 0.0

class NightStreetSUPGRecallQuery(tasti.SUPGRecallQuery):
    # def score(self, target_dnn_output):
    #     return 1.0 if len(target_dnn_output) > 0 else 0.0
    def score(self, results):
        # print('[results]: ',results)
        if(len(results) ==0):
            return 0
        else:
            max_confidence = results[0]['confidence']
            return 1.0 if max_confidence > 0 else 0.0

class NightStreetLHSPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)

class NightStreetLHSRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)
