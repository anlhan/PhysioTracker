package com.anl.physiotracker;

import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    //region Nesne tanımlamaları.
    private static final String TAG = "MainActivity";
    private SensorManager sensorManager;        //Sensör yöneticisi nesne.
    Sensor accelerometer;                       //Sensör nesnesi.
    float[] sensorValues = new float[3];        //3 farklı yöndeki ivme değerlerinin toplanacağı dizi.

    Button recordButton;                        //Kayıt butonu.
    ImageView indicatorImageView;               //Kaydı gösteren yuvarlak gösterge.
    TextView countdownIndicator;                //En alttaki veri toplanma aşamasının takibini sağlayan gösterge.
    TextView results;                           //Tahminlerin üst üste eklendiği textview.
    TextView predString;                        //Classifier'ın tahmin ettiği hareket tipi.

    private Handler handler = new Handler();    //Veri toplayacak Runnable için Handler.
    List<AccDataClass> accVerileri;             //Sensör verilerinin toplanacağı liste.
    AccDataClass yeniVeri;                      //Yeni gelen veri için nesne.
    float meanOfAxisValues[];                   // Sensör verilerinin ortalamaları.
    float stdsOfAxisValues[];                   // Sensör verilerinin standart sapmaları.

    int veriPeriyot = 30;                       //Default 30ms periyot Handler için.
    int veriMiktari = 100;                      //Default 100 defa accelerometer verisi alınacak. Default olarak 100x30ms = 3 saniye.
    Instances actRecognitionDataset;            //Verisetini alacak nesne.
    NaiveBayes naiveBayes;                      //Classifier nesnesi.
    //endregion

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d(TAG, "onCreate: Sensor servisi başlatılıyor...");
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorManager.registerListener(MainActivity.this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        Log.d(TAG, "onCreate: İvmeölçer başlatıldı.");

        recordButton = findViewById(R.id.buttonRecord);
        indicatorImageView = findViewById(R.id.indicator);
        countdownIndicator = findViewById(R.id.countdownTextView);
        results = findViewById(R.id.resultsTextView);
        results.setMovementMethod(new ScrollingMovementMethod());
        predString = findViewById(R.id.predictionValue);
        accVerileri = new ArrayList<>();

        //region Veri seti okunuyor ve classifier oluşturuluyor.
        try {
            InputStream inpStream = getResources().getAssets().open("ActRecDataSet.arff");
            DataSource source = new DataSource(inpStream);          //assets klasöründeki data set dosyamızın input streami kullanılarak veri kaynağı oluşturuluyor.
            actRecognitionDataset = source.getDataSet(); //Veri setindeki değerleri temsil eden Instances nesnesi.

            actRecognitionDataset.setClassIndex(actRecognitionDataset.numAttributes() - 1); // Class değerinin hangi attribute olduğu kararlaştırılıyor.
            naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(actRecognitionDataset);
            inpStream.close();

            // results.append(actRecognitionDataset.relationName()); //Test amaçlı ilişki adını loglamak için.

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        //endregion

    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        //Sensör verileri değişim durumunda sensorValues dizisine kayıt ediliyor.
        //Böylece güncel veri sürekli elimizde.
        sensorValues[0] = sensorEvent.values[0];
        sensorValues[1] = sensorEvent.values[1];
        sensorValues[2] = sensorEvent.values[2];
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    public void onClickRecordButton(View view) {

        handler.postDelayed(recognitionTask, veriPeriyot);
        indicatorImageView.setColorFilter(Color.RED);
        view.setEnabled(false);

    }

    public void onClickResetButton(View view) {

        results.setText("");
        countdownIndicator.setText("--");
        indicatorImageView.setColorFilter(Color.WHITE);
        recordButton.setEnabled(true);
        predString.setText("Unknown");
        handler.removeCallbacks(recognitionTask);
        accVerileri.clear();
        dataCounter = 0;

    }

    public int dataCounter = 0; // 100 sensör verisiyle sınırlamak için sayac.

    //region Runnable oluşturuluyor.
    private Runnable recognitionTask = new Runnable() {

        @Override
        public void run() {
            if (dataCounter == veriMiktari) {                   //Veri alımını sınırlamak için.

                //Nitelik çıkarımı kısmı.Ortalama ve standart sapmalar.
                meanOfAxisValues = meanCalculate();
                stdsOfAxisValues = stdCalculate();

                Instance newInput = new DenseInstance(7);   //Alınan verilerden oluşturulan nesne.
                newInput.setValue(0, meanOfAxisValues[0]);
                newInput.setValue(1, meanOfAxisValues[1]);
                newInput.setValue(2, meanOfAxisValues[2]);
                newInput.setValue(3, stdsOfAxisValues[0]);
                newInput.setValue(4, stdsOfAxisValues[1]);
                newInput.setValue(5, stdsOfAxisValues[2]);

                newInput.setDataset(actRecognitionDataset);             //Oluşturulan nesnenin dataset tanımlaması.

                try {      //Tahmin bloğu. Tahmin edilen double değeri eşleşen aktivite stringine dönüştürülüp basılıyor.
                    double predicted = naiveBayes.classifyInstance(newInput);
                    String predictionString = actRecognitionDataset.classAttribute().value((int) predicted);

                    //Burada timestamp eklenebilir.
                    SimpleDateFormat dateFormat = new SimpleDateFormat("dd.MM.yyyy HH:mm:ss");
                    String millisInString  = dateFormat.format(new Date());

                    results.append("\nTahmin : " + predictionString + "  " + millisInString);
                    predString.setText(predictionString);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                dataCounter = 0;
                accVerileri.clear();
                //return; //bu return 100 veri ile sınırlamak amaçlı. Veri seti toplarken yoruma alındı. Gerekli değil?
            }

            if (dataCounter % 10 == 0) //Her 10 veri alışta bir, gösterge olarak kullandığım textview'ı günceller.
                countdownIndicator.setText(dataCounter + "");

            yeniVeri = new AccDataClass();      //yeni sensör verisi.
            yeniVeri.xVeri = sensorValues[0];
            yeniVeri.yVeri = sensorValues[1];
            yeniVeri.zVeri = sensorValues[2];
            accVerileri.add(yeniVeri);          //listeye veri ekleniyor.
            dataCounter++;

            handler.postDelayed(this, veriPeriyot); //iç içe çağrı için.
        }
    };
    //endregion

    public float[] meanCalculate() {
        int size = accVerileri.size();
        float results[] = {0f, 0f, 0f};
        for (AccDataClass a : accVerileri) {
            results[0] += a.xVeri;
            results[1] += a.yVeri;
            results[2] += a.zVeri;
        }

        results[0] = results[0] / size;
        results[1] = results[1] / size;
        results[2] = results[2] / size;

        return results;
    }

    public float[] stdCalculate() {
        int size = accVerileri.size();
        float results[] = {0f, 0f, 0f};

        for (AccDataClass a : accVerileri) {
            results[0] += Math.pow(a.xVeri - meanOfAxisValues[0], 2);
            results[1] += Math.pow(a.yVeri - meanOfAxisValues[1], 2);
            results[2] += Math.pow(a.zVeri - meanOfAxisValues[2], 2);
        }

        results[0] = (float) Math.sqrt(results[0] / size);
        results[1] = (float) Math.sqrt(results[1] / size);
        results[2] = (float) Math.sqrt(results[2] / size);

        return results;
    }
}