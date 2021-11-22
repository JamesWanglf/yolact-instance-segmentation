import requests

if __name__ == '__main__':
    for i in range(1):
        res = requests.get(
            'http://localhost:6337/detect_objects?image_path=james_test_image_1.jpg&top_k=8&score_threshold=0.5'
            # 'http://localhost:6337/init_engine?model=yolact_resnet50_54_800000.pth'
        )

        print(res.status_code)
        print(res.text)
