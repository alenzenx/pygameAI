# 這是用pycuda 接 cuda c 寫的底層cnn 部分參考https://www.zhihu.com/question/22298352

包含 : 

       convolution層 

       softmax層 
     
       backpropagation(反向傳播)

convolution層 : cuda處理

softmax層 : numpy處理

本版本只有先訓練 手遊:世界計畫 的4block跟6block的分類而已 

有想進一步寫ai外掛程式的歡迎私訊我 我開共同編輯給你(我目前沒空完成)

如果有大神有更好的見解 程式技術部分或演算法... 也歡迎指教

要先裝 : cuda cudnn 以及 vs的c++編譯器，我的環境包:

https://drive.google.com/drive/folders/1LS0JcdhS3sYGMSDsXbELmDhKfCHJePbT?usp=sharing