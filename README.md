# hobby

## Quick start

`cnn+lstm.py`: cnn + lstm으로 train

`cnn.py`: cnn으로만 trian
 
## custom 데이터 수집 및 데이터 전처리

### 1. Video 데이터 수집

1. `videos/video.py`: 웹캠으로 부터 숫자 Class로 비디오를 저장

2. `videos/video_alphbet.py`: 웹캠으로 부터 알파벳 Class로 비디오를 저장

프로그램을 실행후 위와 같은 화면이 나오면 자신이 만들고 싶은 Class에 해당하는 키를 눌러 녹화를 시작하고 다시 똑같은 키를 눌러 녹화를 저장합니다. (<span style="color:red"> 웹캠프로그램이 맨위에 있어야함 </span>)

> 예) a에 대한 영상을 얻으려면 a눌러 녹화를 시작하고 a를 적었으면 다시 a를 눌러 녹화를 저장합니다. 

### 2. Video 데이터에서 Keypoint 수집하기
* Openpose를 설치해둬야함 [설치법](https://juhwakkim.github.io/2019/04/08/2019-04-08-Openpose-tutorial/#more) 

1. `Openpose/04_keypoints_hand_from_video.py`: 저장된 비디오로 부터 숫자 Class에 관한 Keypoint를 추출하여 CSV형태로 저장합니다.

2. `Openpose/04_keypoints_hand_from_video_alphabet.py`: 저장된 비디오로 부터 알파벳 Class에 관한 Keypoint를 추출하여 CSV형태로 저장합니다.

CSV파일은 `Video/{해당하는 class}` 안에  저장됩니다.

### 3. 데이터 전처리

1. `Centered.py`: 저장된 Keypoint의 좌표를 중앙으로 이동시키고 속도를 위해 pickle 형식으로 저장(`/DATA/Centered`에 저장됨)

2. `augmentation.py`: Gan을 제외한 augmentation 적용하는 프로그램(설정에 따라 augmentation을 얼마큼 할지 정할 수 있음)

``` python
    ...

    for i in range(size * 1):
        a = random.randint(0,3)

        A = str(i//(size/10))
        B = str(i%(size/10))

    ...
```
에서 `size * 1` 에서 뒤 상수만큼 상수배 만큼 augmentation이 됩니다.

### 4. imageloader.py

`imageloader.py`: 데이터를 불러오고 이미지를 만드는 프로그램

#### 사용법
``` python
from imageloader import data_process # import

number = False # True면 숫자 False면 알파벳으로 설정
dp_train = data_process('./DATA/aug/all/train/Alphabet',number) # 데이터의 주소값과 위 number 변수를 넣어줍니다.
dp_train.point_data_load() # point 데이터를 불러옵니다
dp_train.sequence_64() # point data를 sequence를 64개로 만든다.
dp_train.image_make() # 이미지를 만들어냅니다. (DATA/image에 저장됨)
dp_train.data_shuffle() # 데이터를 Random으로 shuffle 해줍니다.

dp_train.point # point data(numpy array).
dp_train.images # image data
dp_train.label # label data

```

## Training

`cnn+lstm.py`: cnn + lstm으로 train

`cnn.py`: cnn으로만 trian