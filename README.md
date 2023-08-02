## 研究架構圖
![PAPER - complete process](https://github.com/110598071/Driver_Behavior_Recognition/assets/88961674/249f0d6c-f555-4596-9f1e-202669622fa1)
---
## Repo建置
1. clone OpenPose PytTorch版
    - https://github.com/110598071/pytorch-openpose

2. clone 此Repo(含有兩個資料夾，後續將詳細說明)
    - https://github.com/110598071/Driver_Behavior_Recognition

3. 下載AUC Dataset及Milestone Model
    - 將dataset及model資料夾放在/Driver_Behavior_Recognition/experiment_20230211/
    - XXX

Step 4：環境建置
1. GPU：2080Ti
2. PyTorch 1.12.0+cu113
    ```pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio===0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```
4. CUDA 11.3
5. CUDNN 8.5.0
6. download relative package
    ```pip install -r requirements.txt```
    
---
## 檔案及資料夾結構說明

Repo中含有兩個命名開頭為experiment的資料夾
分別為兩段時期所做的實驗記錄
多數論文實驗成果皆以experiment_20230211紀錄之
註：有些檔案命名或程式碼會怪怪的或有點亂，請見諒

- experiment_20230211資料夾結構
    ```
    ── experiment_20230211
     ├ csv
     ├ dataset
     ├ model
     ├ plot
     ├ python
     └ diary.xlsx
    ```

- diary.xlsx
    - 紀錄所有的模型成果，包含評估指標、訓練時間戳等各種資訊
    - 各工作表分別為：
        - CNN：Driver Image Feature branch之CNN
        - CNN_ablation：CNN消融實驗
        - one cycle lr：CNN之learning rate設定
        - Fast RCNN：Driver Behavior Feature branch之Faster RCNN
        - concatenate：Feature Concatenate branch之perceptron
        - perceptron_ablation：perceptron消融實驗
- csv
    - 各csv為資料集影像輸入OpenPose後所取得的人體骨骼資料
        - 資料包含影像名稱、各部位之座標(xy座標)、信心分數、偵測到的部位數量
    - 沒有數字後綴詞的csv檔皆為使用原始影像作為輸入，反之，則影像將調整為對應解析度後才輸入至OpenPose進行偵測
        - ex. AUC_skeleton_merge_test_data_400係將影像調整為400*400再輸入至OpenPose進行偵測，因此各部位之座標點最大值為400
- dataset
    - 資料集架構
      ![PAPER - Page 21](https://github.com/110598071/Driver_Behavior_Recognition/assets/88961674/9f111c0e-9d0b-4283-a02f-5578fd324f6d)
    - 資料夾結構
    ```
    ── dataset
     ├ AUC
     | ├ Camera 1
     | | ├ test
     | | └ train
     | └ Camera 2
     |   ├ test
     |   └ train
     ├ AUC_empty_template
     ├ AUC_output
     ├ AUC_processed
     ├ AUC_processed_merge
     ├ AUC_processed_merge_data_balance
     ├ AUC_processed_merge_data_over_sampling
     ├ AUC_processed_merge_lableImg
     ├ AUC_processed_merge_rcnn_mask
     ├ AUC_processed_merge_wrong_classification
     ├ AUC_remove
     ├ CONCATE_DATASET_CSV
     ├ object_detection_test
     ├ StateFarm
     ├ initial_weight.csv
     ├ misclassification analysis.xlsx
     ├ object_feature_20230428_0445.xlsx
     └ Test_Dataset_Concate_Features.xlsx
    ```
    - 資料夾說明
        - AUC：原始資料集，分為Camera 1/2(左右駕影像)，並切分為訓練/測試集
        - AUC_empty_template：結構同AUC資料夾，但沒有任何檔案，資料移動時會用到
        - AUC_output：將資料集輸入OpenPose並取得繪製骨骼的輸出影像
        - AUC_processed：移除OpenPose無法辨識的影像(這個流程其實是錯的，未來要修正)
        - AUC_processed_merge：將AUC_processed的Camera 1/2影像合併
        - AUC_processed_merge_data_balance：手動針對c0類別進行欠採樣(移除重複過高的影像，效果不彰因此沒寫進論文內)
        - AUC_processed_merge_data_over_sampling：手動針對c1-c9類別進行過採樣(複製較為困難或標誌性的影像，效果不彰因此沒寫進論文內)
        - AUC_processed_merge_lableImg：存放手工物件label的標註資料
        - AUC_processed_merge_rcnn_mask：使用Faster RCNN進行駕駛人遮罩處理後的影像(亦即CNN的輸入影像)
        - AUC_processed_merge_wrong_classification：各模型的錯誤分類影像，可用於評估模型
        - AUC_remove：存放AUC_processed中所移除的影像
        - CONCATE_DATASET_CSV：存放影像經過前兩個分支後的資料，也就是輸入perceptron前的資料，因為這些資料的取得需要經過Faster RCNN的偵測以及CNN的卷積，大約需要20分鐘，如果日後需要調整perceptron的架構，每次都需要重新取得資料的話會花很多時間，因此可以先儲存起來，在相同Faster RCNN和CNN模型的情況下去調整perceptron架構
        - object_detection_test：存放Faster RCNN模型訓練資料
        - StateFarm：StateFarm資料集
        - initial_weight.csv：用來記錄CNN初始權重(當時只是用來確認一些資訊)
        - misclassification analysis.xlsx：用於紀錄錯誤分類影像(當時只是用來評估模型)
        - object_feature_20230428_0445.xlsx：用來評估模型的產物，我也忘了是評估啥
        - Test_Dataset_Concate_Features.xlsx：用來評估模型的產物，我也忘了是評估啥

- model
    - 資料集結構
    ```
    ── model
     ├ cnn
     └ fast rcnn
    ```
    - 資料夾說明
        - cnn：存放CNN以及Perceptron model
        - fast rcnn：存放Faster RCNN model
    - 各模型皆可使用時間戳於diary.xlsx蒐尋其模型訓練結果
- plot
    - 資料集結構
    ```
    ── plot
     ├ Accuracy
     ├ box_plot
     ├ Confusion
     ├ fast_rcnn_loss
     ├ feature_map
     ├ Loss
     └ 測試FC架構
    ```
    - 資料夾說明
        - Accuracy：存放各模型的accuracy plot
        - box_plot：繪製feature box plot，用於分析
        - Confusion：存放各模型的混淆矩陣，分為百分比模式以及數量模式
        - fast_rcnn_loss：存放Faster RCNN的loss plot
        - feature_map：某次meeting用於報告的圖，useless
        - Loss：存放CNN以及Perceptron的loss plot
        - 測試FC架構：老實說我也忘了這是在幹嘛
    - 各檔案命名與模型之時間戳對應
- python
    - 資料集結構
    ```
    ── python
     ├ .idea
     ├ __pycache__
     ├ fast_rcnn
     | ├ __pycache__
     | ├ config.py
     | ├ datasets.py
     | ├ engine.py
     | ├ feature_computation.py
     | ├ inference.py
     | ├ IoU_computation.py
     | ├ mAP_computation.py
     | ├ model.py
     | └ utils.py
     ├ AlexNet.py
     ├ backup.py
     ├ FCLayer.py
     ├ feature_dataset.py
     ├ get_skeleton_img_and_csv.py
     ├ hyperparameters_opt.py
     ├ Image_dataset.py
     ├ PSO.py
     ├ pytorch_config.py
     ├ pytorch_model.py
     ├ pytorch_progress.py
     ├ pytorch_train_model.py
     ├ pytorch_util.py
     ├ pytorch_voting.py
     ├ skeleton_util.py
     ├ test.py
     └ train_with_openpose.py
    ```
    - 檔案說明
        - fast_rcnn/config.py：Faster RCNN訓練時所使用的超參數設定
        - fast_rcnn/datasets.py：Faster RCNN訓練時所使用的Dataset
            - ObjectDataset：使用手工label的標註資料轉換為dataset
            - InferenceDataset：Pseudo-Labeling使用model偵測出的標註資料轉換為dataset
            - train_loader：Faster RCNN訓練集
            - valid_loader：Faster RCNN測試集
        - fast_rcnn/engine.py：Faster RCNN模型訓練主程式
            - simple_RCNN_training：只訓練一次
            - semi_Supervised_Learning：使用Pseudo-Labeling進行半監督式學習
        - fast_rcnn/feature_computation.py：使用Faster RCNN模型偵測出物件資料，並以此進行Driver Behavior Feature的計算
            - ImageObjectDetection：用於單一影像的特徵計算
            - feature_computation：用於整個資料集影像的特徵計算
            - normalize_final_features：針對資料進行標準化
        - fast_rcnn/inference.py：使用訓練完成的Faster RCNN模型進行影像物件偵測
        - fast_rcnn/IoU_computation.py：使用訓練完成的Faster RCNN模型進行IoU評估
        - fast_rcnn/mAP_computation.py：使用訓練完成的Faster RCNN模型進行mAP評估
        - fast_rcnn/model.py：用於load Faster RCNN pretrained model
        - fast_rcnn/utils.py：存放Faster RCNN模型訓練過程中會重複使用的函數
        - AlexNet.py：用於load AlexNet pretrained model(我也忘了為啥會獨立出來)
        - backup.py：存放備份程式碼，useless
        - FCLayer.py：存放各pretrained model的全連接層(包含手工設計的)
        - feature_dataset.py：CNN訓練時所使用的Dataset
        - get_skeleton_img_and_csv.py：使用OpenPose偵測影像之人體骨骼並取得輸出骨骼繪製影像以及csv輸出資料
        - hyperparameters_opt.py：CNN和Perceptron的超參數最佳化
            - CNN_hyperparameter_optimization：CNN超參數最佳化
            - SLP_MLP_hyperparameter_optimization：Perceptron超參數最佳化
        - Image_dataset.py：CNN訓練時所使用的Dataset以及影像前處理函數(ex. blur、hog)
        - PSO.py：粒子群演算法超參數最佳化(論文沒用到，未來可嘗試)
        - pytorch_config.py：Perceptron模型訓練的前置處理
        - pytorch_train_model.py：用於load CNN pretrained model
        - pytorch_util.py：存放CNN及Perceptron模型訓練過程中會重複使用到的函數
        - pytorch_voting.py：曾經嘗試使用voting方式進行concate但成效不彰，useless
        - skeleton_util.py：用於OpenPose偵測人體骨骼時將部位資料轉換為特徵資料
        - test.py：用於測試各種天馬行空的想法，也留了許多腳印，或許可以參考一下
        - train_with_openpose.py：單純使用OpenPose偵測到的部位資料轉換為特徵資料再進行模型訓練
