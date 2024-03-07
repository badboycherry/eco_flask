from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
import threading
import easyocr

app = Flask(__name__)

# OCR 모델 초기화
reader = easyocr.Reader(['ko'])

# OCR 엔드포인트 정의
@app.route('/ocr', methods=['POST'])
def ocr():
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

        # OCR 수행
        result = reader.readtext(img_path)

        # 등록 성공 여부를 나타내는 변수
        registration_success = False

        # OCR 결과를 검사하여 "친환경"과 "녹색인증" 단어가 포함되어 있는지 확인
        for detection in result:
            text = detection[1]
            if "친환경" in text or "녹색인증" in text or "저탄소" in text:
                registration_success = True
                break

        # 등록 결과 출력
        if registration_success:
            print("등록 성공")
        else:
            print("등록 실패")

        return jsonify({'registration_success': registration_success})

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
