#include "detector.h"

float IDetector::Iou(const std::vector<float> &boxA, const std::vector<float> &boxB)
{
    const float eps = 1e-6;
    float area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    float area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    float x1 = std::max(boxA[0], boxB[0]);
    float y1 = std::max(boxA[1], boxB[1]);
    float x2 = std::min(boxA[2], boxB[2]);
    float y2 = std::min(boxA[3], boxB[3]);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    return inter / (area_a + area_b - inter + eps);
}

void IDetector::Nms(std::vector<std::vector<float>> &boxes, const float iou_threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<float> &boxA, const std::vector<float> &boxB) { return boxA[4] > boxB[4]; });
    for (int i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i][4] == 0.f)
            continue;
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (boxes[i][5] != boxes[j][5])
            {
                continue;
            }
            if (Iou(boxes[i], boxes[j]) > iou_threshold) {
                boxes[j][4] = 0.f;
            }
        }
    }
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const std::vector<float> &box) {return box[4]==0.f;}),boxes.end());
}

void IDetector::BoundariesLogic(std::vector<std::vector<float>> &boxes)
{
    for (int i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i][4] == 0.f)
            continue;
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (boxes[i][5] != boxes[j][5])
            {
                if ((boxes[i][5]==2) || (boxes[j][5]==2)) {
                    continue;
                }
                else {
                    if (Iou(boxes[i], boxes[j]) > 0.05) {
                        if (boxes[i][5] == 1) {
                            boxes[j][4] = 0.f;
                        }
                        if (boxes[j][5] == 1) {
                            boxes[i][4] = 0.f;
                        }
                        continue;
                    }
                }
            }
        }
    }
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const std::vector<float> &box) {return box[4]==0.f;}),boxes.end());
}

int IDetector::DetectionLogic(std::vector<std::vector<float>> &boxes)
{
	int empty_chairs=0;
	std::vector<std::vector<float>>chairs;
	std::vector<std::vector<float>>people;
	for(const auto &box:boxes)
	{
		if(box[5] == 2)
		{
			chairs.push_back(box);
		}
		else if(box[5] == 1)
		{
			people.push_back(box);
		}
		for(const auto &chair :chairs)
		{
			bool is_empty = true;
			for(const auto &person : people)
			{
				if(Iou(chair,person)>0.05)
				{
					is_empty = false;
					break;
				}
			}
			if(is_empty)
				empty_chairs++;
		}
		return empty_chairs;
	}
}
