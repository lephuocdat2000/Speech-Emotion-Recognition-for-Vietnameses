# Speech-Emotion-Recognition-for-Vietnameses
Nhận diện cảm xúc trong câu nói tiếng Việt

Input: Một câu nói Tiếng Việt khoảng 5-10 câu

Output: Cảm xúc của câu nói đó ( Angry, Happy, Sad, Neutral

Feature Engineering: Sử dụng đặc trưng mel-spectrogram là đặc trưng biển diễn âm thanh dưới dạng tần số

Kiến trúc mạng: 
  - TimedistributedCNN : 

        + Gồm 4 khối local feature learning block nối tiếp nhau​

        + Mỗi khối bao gồm: 

                  - Conv2D​

                  - BatchNormalization

                  - Activation

                  - MaxPooling

                  - Dropout

   - LSTM : gồm 256 units

   - Dense: Activation = softmax
 
Đánh giá mô hình : F1-score: 0.72

Run app.py -> Use link http://127.0.0.1:5000/ to try
