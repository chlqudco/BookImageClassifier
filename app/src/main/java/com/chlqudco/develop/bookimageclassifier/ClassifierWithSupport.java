package com.chlqudco.develop.bookimageclassifier;

import static org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod.NEAREST_NEIGHBOR;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;

public class ClassifierWithSupport {
    private static final String MODEL_NAME = "mobilenet_imagenet_model.tflite";
    private static final String LABEL_FILE = "labels.txt";

    //모델 관련 정보들
    int modelInputWidth, modelInputHeight, modelInputChannel;
    
    //라벨 파일을 읽어서 클래스명을 저장할 리스트
    private List<String> labels;

    //모델에 쳐 넣기 전까지 모든 과정에서 이미지 및 관련 데이터 갖고 있을 변수
    TensorImage inputImage;

    //인터프리터
    Interpreter interpreter;

    //출력값 담을 변수, 출력 결과값을 효율적으로 다룰 수 있음
    TensorBuffer outputBuffer;

    //메인에서 가져올 콘텍스트
    Context context;

    //생성자로 콘텍스트 초기화
    public ClassifierWithSupport(Context context) {
        this.context = context;
    }

    //초기화
    public void init() throws IOException{
        // loadMappedFile 메서드로 손쉽게 바이트버퍼형으로 변환 가능, loadModelFile함수 안써도 됨
        ByteBuffer model = FileUtil.loadMappedFile(context, MODEL_NAME);
        model.order(ByteOrder.nativeOrder());
        interpreter = new Interpreter(model);

        initModelShape();
        
        //파일을 읽어서 클래스들을 리스트 형태로 자동 저장
        labels = FileUtil.loadLabels(context, LABEL_FILE);
    }

    //모델 이미지 전처리
    private void initModelShape() {
        Tensor inputTensor = interpreter.getInputTensor(0);
        int[] inputShape = inputTensor.shape();
        modelInputChannel = inputShape[0];
        modelInputWidth = inputShape[1];
        modelInputHeight = inputShape[2];

        //모델의 입력값의 데이터 타입을 생성자에 전달해서 모델과 동일한 타입으로 데이터 타입 설정
        inputImage = new TensorImage(inputTensor.dataType());

        Tensor outputTensor = interpreter.getOutputTensor(0);

        //결과값 받아올 변수 초기화
        outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType());
    }

    //뭐야이건
    private Bitmap convertBitmapToARGB8888(Bitmap bitmap) {
        return bitmap.copy(Bitmap.Config.ARGB_8888,true);
    }

    //Bitmap이미지를 전처리 하고 TensorImage 형태로 반환하는 함수
    private TensorImage loadImage(final Bitmap bitmap){
        //이미지 저장
        if(bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            inputImage.load(convertBitmapToARGB8888(bitmap));
        } else {
            inputImage.load(bitmap);
        }

        //이미지 전처리 해주는 도구 생성
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                        //이미지 크기는 어떻게 만들건지
                .add(new ResizeOp(modelInputWidth, modelInputHeight, NEAREST_NEIGHBOR))
                        //이미지 정규화 어떻게 할건지
                .add(new NormalizeOp(0.0f, 255.0f))
                .build();

        //프로세서에 우리의 이미지를 넣고 전처리한 결과값 반환(텐서이미지 형식)
        return imageProcessor.process(inputImage);
    }

    public Pair<String, Float> classify(Bitmap image){
        //이미지 전처리
        inputImage = loadImage(image);

        //추론 실행
        interpreter.run(inputImage.getBuffer(), outputBuffer.getBuffer().rewind());
        
        //출력값과 클래스 이름 매핑
        Map<String, Float> output = new TensorLabel(labels, outputBuffer).getMapWithFloatValue();
        
        //최대값 반환
        return argmax(output);
    }

    //클래스와 확률중 최대확률 반환
    private Pair<String, Float> argmax(Map<String, Float> map) {
        String maxKey = "";
        float maxVal = -1;

        for(Map.Entry<String, Float> entry : map.entrySet()) {
            float f = entry.getValue();
            if(f > maxVal) {
                maxKey = entry.getKey();
                maxVal = f;
            }
        }

        return new Pair<>(maxKey, maxVal);
    }

    //끝낼때 자원 반환
    public void finish() {
        if(interpreter != null)
            interpreter.close();
    }
}
