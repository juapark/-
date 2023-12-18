from flask import Flask, render_template, request
from ultralytics import YOLO
import numpy as np
from PIL import Image

app = Flask(__name__)

# 현재 작업하는 프로젝트 경로
path = '/Users/parkjua/PycharmProjects/pythonProject1/2023-hanium/'

# 메인 페이지 실행하는 함수
@app.route('/')
def main():
    return render_template('main.html')

def yolo_predict(filename):
    # pretrain 된 모델 불러오기
    model = YOLO(path + 'model/best.pt')

    # 불러온 모델을 통해 이미지 감지
    results = model.predict(path + 'static/images/'+filename)

    head_count = 0
    helmet_count = 0

    for result in results:
        for cls in result.names:
            if cls == 0:
                head_count += 1
            elif cls == 1:
                helmet_count += 1

    # plot을 결과로 저장
    plots = results[0].plot()

    # 이미지 객체 생성
    image = Image.fromarray(plots)

    # 이미지 파일로 저장
    output_filename = path + 'static/output_images/' + filename + '_output.png'
    image.save(output_filename)


    # 감지 결과 사진 리턴
    return filename + '_output.png', head_count, helmet_count


# 결과 감지 페이지 실행하는 함수
@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        # 업로드한 이미지 파일
        f = request.files['file']

        # 이미지 파일을 저장할 경로 설정
        f_name = path + 'static/images/' + f.filename

        # 이미지 저장
        f.save(f_name)

        # yolov8 모델을 통해 감지한 결과 이미지
        output_filename, head_count, helmet_count = yolo_predict(f.filename)

        # 원본 이미지와 결과 이미지를 리턴
        return render_template('result.html', filename=['/images/'+f.filename], output_image=['/output_images/'+output_filename], helmet_count=helmet_count, head_count=head_count)
    else:
        return render_template('main.html')

@app.route('/history')
def history():
    # 헬멧 감지 이력 페이지를 렌더링합니다.
    return render_template('history.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 폼에서 전송된 데이터 가져오기
        name = request.form.get('name')
        birth = request.form.get('birth')
        gender = request.form.get('gender')
        position = request.form.get('position')
        return render_template('register_complete.html', name=name, birth=birth, gender=gender, position=position)
    else:
        return render_template('register.html')

if __name__ == '__main__':
    app.run(port="9999", debug=True)


