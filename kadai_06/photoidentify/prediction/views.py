from django.shortcuts import render
from .forms import ImageUploadForm
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import load_img  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import img_to_array  # pyright: ignore[reportMissingImports]
from io import BytesIO
import os
from django.conf import settings
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions  # pyright: ignore[reportMissingImports]

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # バッチ次元を追加
            # img_array = img_array.reshape((1, 224, 224, 3))
            # img_array = img_array/224
            # VGG16モデルに合わせた前処理
            img_array = preprocess_input(img_array)
            
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            # result = model.predict(img_array)
            
            # 推論を実行
            preds = model.predict(img_array)
            
            # 推論結果をデコード
            decoded_preds = decode_predictions(preds, top=5)[0] # 上位5件を取得
            
            # 辞書形式で結果を格納
            # VGG16モデルの出力は、1000個のImageNetクラスです
            # `decode_predictions`は、クラスID、クラス名、確率のタプルをリストで返します
            prediction = []
            for i, (imagenet_id, label, prob) in enumerate(decoded_preds):
                prediction.append({
                    'category': label,
                    'probability': float(prob)
                })

            return render(request, 'home.html', {'form': form, 'prediction': prediction})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
            
            # if result[0][0] > result[0][1]:
            #     prediction = '猫'
            # else:
            #     prediction = '犬'
            # return render(request, 'home.html', {'form': form, 'prediction': prediction})
        # else:
            # form = ImageUploadForm()
            # return render(request, 'home.html', {'form': form})