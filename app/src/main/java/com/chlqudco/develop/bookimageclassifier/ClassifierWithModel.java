package com.chlqudco.develop.bookimageclassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;
import android.util.Size;

import androidx.annotation.Nullable;

import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod.NEAREST_NEIGHBOR;

//더욱? 간소화한 클래스(Model클래스 이용), 인터프리터 할일, tflite 파일 로드 를 모두 Model 클래스로 처리
public class ClassifierWithModel{
    private static final String MODEL_NAME = "mobilenet_imagenet_model.tflite";
    private static final String LABEL_FILE = "labels.txt";

    Context context;

    //젤 중요한 모델 클래스 변수 생성
    Model model;

    int modelInputWidth, modelInputHeight, modelInputChannel;

    TensorImage inputImage;

    TensorBuffer outputBuffer;

    private List<String> labels;

    private boolean isInitialized = false;

    public ClassifierWithModel(Context context) {
        this.context = context;
    }

    public void init() throws IOException {
        //모델 불러와서 저장
        model = Model.createModel(context, MODEL_NAME);

        initModelShape();
        labels = FileUtil.loadLabels(context, LABEL_FILE);

        isInitialized = true;
    }

    public boolean isInitialized() {
        return isInitialized;
    }

    private void initModelShape() {
        //모델을 이용해 입력 텐서 하나 가져오기
        Tensor inputTensor = model.getInputTensor(0);

        int[] shape = inputTensor.shape();
        modelInputChannel = shape[0];
        modelInputWidth = shape[1];
        modelInputHeight = shape[2];

        inputImage = new TensorImage(inputTensor.dataType());

        //모델을 이용해 출력 텐서 가졍기
        Tensor outputTensor = model.getOutputTensor(0);
        outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType());
    }

    public Size getModelInputSize() {
        if(!isInitialized)
            return new Size(0, 0);
        return new Size(modelInputWidth, modelInputHeight);
    }

    //텐서이미지 클래스는 ARGB_8888 형식의 비트맵만 입력받으므로 변환
    private Bitmap convertBitmapToARGB8888(Bitmap bitmap) {
        return bitmap.copy(Bitmap.Config.ARGB_8888,true);
    }

    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        if(bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            inputImage.load(convertBitmapToARGB8888(bitmap));
        } else {
            inputImage.load(bitmap);
        }

        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(modelInputWidth, modelInputHeight, NEAREST_NEIGHBOR))
                .add(new Rot90Op(numRotation))
                .add(new NormalizeOp(0.0f, 255.0f))
                .build();

        return imageProcessor.process(inputImage);
    }

    //추론
    public Pair<String, Float> classify(Bitmap image, int sensorOrientation) {
        inputImage = loadImage(image, sensorOrientation);

        //run 메소드에 넣을 값들 정리
        Object[] inputs = new Object[]{inputImage.getBuffer()};
        Map<Integer, Object> outputs = new HashMap();
        outputs.put(0, outputBuffer.getBuffer().rewind());

        //인터프리터가 안하고 모델이 추론함
        model.run(inputs, outputs);

        Map<String, Float> output = new TensorLabel(labels, outputBuffer).getMapWithFloatValue();

        return argmax(output);
    }

    public Pair<String, Float> classify(Bitmap image) {
        return classify(image, 0);
    }

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

    public void finish() {
        if(model != null) {
            model.close();
        }
    }
}
