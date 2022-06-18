# LSTM Pose Classification

![pose](https://user-images.githubusercontent.com/81678439/174430629-a8b974d6-6cbf-4cbd-9e5a-17c4fc3446b3.gif)

Mediapipe BlazePose와 LSTM을 이용한 Pose Classification을 진행하며, OpenCV를 활용하여 공을 추적해서
현재 공을 다루는 사람의 자세와, 현제 수행중인 동작을 식별하는 간단한 프로그램이며, 이를 활용하여 최종적으로는 축구 등 스포츠 동작을 구분하여 사용자에게 피드백을 제공 할 수 있는 프로그램을 작성하고자 합니다.

# 로컬 개발 환경 설정

$ ~/pose/python/ 에서

```
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```
```
python data_producer.py
python lstm_model.py
python lstm_pose_classifier.py
```

종료시 ```deactivate```


# Mediapipe Blazepose Keypoints

<img width="748" alt="스크린샷 2022-04-25 오전 11 21 35" src="https://user-images.githubusercontent.com/81678439/165010419-d66da0ee-e537-46bb-a6c1-1134d568225c.png">
