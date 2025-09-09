from flask import Flask, request, send_file, jsonify, send_from_directory
import shutil
from pathlib import Path
import subprocess
import os
from infer_MedSAM2_slicer import perform_inference, improve_inference
import os
from werkzeug.utils import secure_filename
from totalsegmentator.python_api import totalsegmentator
import numpy as np


app = Flask(__name__) # default로 Flask앱생성
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok = True)
os.makedirs(OUTPUT_FOLDER, exist_ok = True)
app.config['INPUT_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# 허용할 파일 확장자를 설정할 수 있습니다 (예: {'nii', 'nii.gz'})
ALLOWED_EXTENSIONS = {'nii', 'gz'}

predictor_state = {} # 예측한 segmentation
def allowed_file(filename):
    """파일 확장자가 허용되는지 확인하는 함수"""
    return '.' in filename and \
           (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or \
            filename.rsplit('.', 2)[1].lower() == 'nii')

###########################################################################################

@app.route('/run_script', methods=['POST'])
def run_script():
    input_name = request.form.get('input') # 입력이미지의 파일경로
    gts_name = request.form.get('gts') # GT 라벨경로
    gts_data = np.load(gts_name)
    segs = gts_data['segs'] # - Total segmentator로 분할된 결과는 잘 나옴 

    #############
    print(f'seg_unique: {np.unique(segs)}')
    #############

    propagate = request.form.get('propagate') in ['y', 'Y'] # Propagation여부(슬라이스하나만 분할할지 아니면 여러개다할지)
    checkpoint = 'checkpoints/%s'%(request.form.get('checkpoint'),) # MedSAM 체크포인트
    cfg = request.form.get('config') # 모델의 설정파일경로

    # INFERENCE
    predictor, inference_state = perform_inference(checkpoint, cfg, input_name, gts_name, propagate, pred_save_dir='data/video/segs_tiny')
    predictor_state['predictor'] = predictor
    predictor_state['inference_state'] = inference_state

    return 'Success'


@app.route('/Totalsegmentator', methods=['POST'])
def segment_image():
    # 1. 요청 유효성 검사
    if 'file' not in request.files:
        return jsonify({"error": "파일이 요청에 포함되지 않았습니다."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "허용되지 않는 파일 형식입니다."}), 400

    # getlist()를 사용하여 'roi_subset'이라는 키를 가진 모든 값을 리스트로 직접 받습니다.
    roi_organs = request.form.getlist('roi_subset')

    # 만약 클라이언트가 roi_subset 파라미터를 아예 보내지 않은 경우,
    # roi_organs는 빈 리스트([])가 됩니다. 이를 None으로 처리해줍니다.
    if not roi_organs:
        roi_organs = None

    print(f"클라이언트로부터 받은 ROI 리스트: {roi_organs}")

    # output_type 파라미터를 받습니다. 기본값은 'nii.gz'
    filetype = request.form.get('output_type', 'nii.gz')

    # 3. 파일 저장 및 경로 설정
    # 안전한 파일 이름으로 변경하여 저장합니다.
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['INPUT_FOLDER'], filename)
    
    # 출력 파일 이름을 정의합니다.
    output_filename = f"segmented_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    file.save(input_path)

    # 4. TotalSegmentator 실행
    try:
        # 로그!
        print(f"TotalSegmentator 실행 시작: {input_path}")
        print(f"ROI: {roi_organs}, Output Type: {filetype}")

        # TotalSegmentator 함수를 호출합니다. ml=True 옵션으로 모든 ROI를 사용 가능하게 합니다.
        # roi_subset이 None이면 모든 가용한 ROI를 분할합니다.
        totalsegmentator(
            input_path, 
            output_path, 
            ml=True, # 이 옵션을 True로 해야 roi_subset 사용이 원활합니다.
            roi_subset=roi_organs,
            output_type=filetype
        )

        
        print(f"TotalSegmentator 실행 완료. 결과: {output_path}")

        # 5. 결과 파일 전송
        # send_from_directory를 사용하여 파일을 클라이언트에게 전송합니다.
        # as_attachment=True는 브라우저에서 바로 열리지 않고 다운로드되도록 합니다.
        return send_from_directory(
            app.config['OUTPUT_FOLDER'],
            output_filename,
            as_attachment=True
        )

    except Exception as e:
        # 오류 발생 시 500 에러와 함께 오류 메시지를 반환합니다.
        return jsonify({"error": f"분할 중 오류 발생: {str(e)}"}), 500
    


# 이거 refine하고 다시 추론할때 이거씀
@app.route('/improve', methods=['POST'])
def improve():
    input_name = request.form.get('input') # input가져오고

    # 그전에 예측한 상태인 predictor_state를 인자로 줘서 결과 반환받음
    predictor, inference_state = improve_inference(input_name, pred_save_dir='data/video/segs_tiny', predictor_state=predictor_state)
    predictor_state['predictor'] = predictor
    predictor_state['inference_state'] = inference_state

    return 'Success'


@app.route('/download_file', methods=['GET'])
def download_file():
    output_name = request.form.get('output')
    return send_file(output_name, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload_file():    
    file = request.files['file']

    if file:
        file.save(file.filename)
        return 'File uploaded successfully'

@app.route('/upload_model', methods=['POST'])
def upload_model():    
    file = request.files['file']
    model_name = os.path.basename(file.filename).split('.')[0]
    checkpoint_dir = "./checkpoints/%s"%model_name

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    file.save(os.path.join(checkpoint_dir, os.path.basename(file.filename)))
    return 'Model uploaded successfully'

@app.route('/upload_config', methods=['POST'])
def upload_config():    
    file = request.files['file']
    config_dir = "./sam2"

    Path(config_dir).mkdir(parents=True, exist_ok=True)

    file.save(os.path.join(config_dir, 'custom_' + os.path.basename(file.filename)))
    return 'Config file uploaded successfully'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
