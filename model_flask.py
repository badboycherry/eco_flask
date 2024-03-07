from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pymysql

app = Flask(__name__)

# 학습된 모델 불러오기
trained_model = load_model('best_model.h5')

# MariaDB 연결 설정
db = pymysql.connect(host='192.168.41.185',
                     user='mk',
                     password='aa5496!!',
                     database='eco',
                     cursorclass=pymysql.cursors.DictCursor)

# 이미지를 VGGNet16 모델에 전처리하는 함수
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 이미지 특징을 추출하는 함수
feature_layer_name = 'dense' 

def extract_features(img_array):
    model = Model(inputs=trained_model.input, outputs=trained_model.get_layer(feature_layer_name).output)
    features = model.predict(img_array)
    return features

# 디비로부터 클래스와 특징을 가져오는 함수
def get_database_features():
    cursor = db.cursor()
    cursor.execute("SELECT class, filename, feature FROM eco_feature")
    result = cursor.fetchall()
    cursor.close()
    return result

def calculate_cosine_similarity(input_feature, db_features):
    similarities = []
    for db_feature in db_features:
        # db_feature['feature']를 부동 소수점 배열로 변환
        db_feature_array = [float(value) for value in db_feature['feature'].strip("[]").split(", ")]
        similarity = cosine_similarity(input_feature, [db_feature_array])

        # 유사도가 0.5 이상인 경우에만 결과 추가
        if similarity[0][0] >= 0.6:
            result_entry = {'class': db_feature['class'], 'filename': db_feature['filename'], 'similarity': similarity[0][0]}
            similarities.append(result_entry)

    return similarities

# /predict 엔드포인트 정의
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 이미지 받아오기
        img_file = request.files['image']
        img_path = 'temp_image.jpg'
        img_file.save(img_path)

        print("이미지가 성공적으로 받아졌습니다.")
        # 이미지 정보 출력
        img_info = {
            'file_name': img_file.filename,
            'content_type': img_file.content_type,
        }
        print("이미지 정보:", img_info)

        # 이미지 전처리 및 특징 추출
        input_img_array = preprocess_image(img_path)
        input_feature = extract_features(input_img_array)

        # 디비로부터 클래스와 특징 가져오기
        db_features = get_database_features()

        # 코사인 유사도 계산
        similarities = calculate_cosine_similarity(input_feature, db_features)

        # 결과 정렬 및 반환
        sorted_results = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

        # 소수점 2째 자리까지 출력하도록 형식 지정
        for result in sorted_results:
            result['similarity'] = round(result['similarity'], 2)

        print("sorted_results:", sorted_results)
        return jsonify(sorted_results)

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)