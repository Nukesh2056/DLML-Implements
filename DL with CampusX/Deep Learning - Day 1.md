
DL is the part of broader family of ML methods based on artificial Neural Networks with [Representation Learning ](https://en.wikipedia.org/wiki/Feature_learning) DL automatically extract the features from the Image Data you provide. 

DL algorithms uses multiple layers to progressively extract higher level feature from the raw input.
For example, In image processing , lower layers may identify edges , while higher layers may identify concepts relevant to human such as digit or letter or faces.  


Starting layers Extract small features like Edges & Textures, Middle & End layers understand the more complex concept of Data.


![[Pasted image 20260309112337.png]]


## Deep Learning Vs Machine Learning 

1. ***Data Dependency*** - DL need more data than ML need the amount of data, You can say Deep Learning is data hungry , if data increases the performance also in DL , in ML it is upto a limit.
2. ***Hardware Dependency*** - DL needs Powerful GPU , because it uses Matrix Multiplication , in ML we can also run the model in Cheap CPU hardware. 
3. ***Training Time*** - DL model can even take week or months to train a DL model , but the prediction is fast in DL, the Training time of ML is very less compared to DL model, uiut hardly takes 1-2 minutes to train a ML model 
4. ***Feature Selection*** - In DL Feature is automatically extracted by the specific DL algorithm you choose , Starting layers extract simple/premative features,  then as layers goes up DL extract more complex features.

![[Pasted image 20260309113341.png]]

5. ***Interpretability/Explainbility*** -  DL can not tell why it chosed the prediction that it made, Eg: If you Deploy the DL model in social media & it bans some some comments of user 1 , suppose user 1 asked why the Fuck his comment is banned by this DL model , DL model can not explain why the comment was banned , I guess this was because the complexity of the model,  BUT In ML models the Explainbility is very high, if the you can tell the person why the comment was banned  ,for eg in linear regression the weight 1 & weight 2 was low that is why comment is banned or something else. 


In some cases the ML performs better than DL 
their is small quote "Jaha pe Sui ki Jarurat hoti hai waha talwar use nahi krte"  , yaha pe sui ML hai aur talwar DL is , depends on your target tumhe kapde silne hai ya logo ko marna hai.



**Deep Learning (jo 1960s se exist karti hai) achanak 2010s ke baad itni famous aur powerful kyun ban gayi?"**

Iske 5 core driving factors hain:

- **1. Massive Datasets [[36:56](http://www.youtube.com/watch?v=fHF22Wxuyw4&t=2216)]:** Deep learning is "data hungry". 2015 ke baad smartphones/cheap internet se exponential data generate hua. Google, Microsoft jaisi companies ne COCO, YouTube-8M, SQuAD jaise huge public datasets open-source kar diye, jisne research ko boost kiya.
    
- **2. Powerful Hardware (GPUs & Beyond) [[43:52](http://www.youtube.com/watch?v=fHF22Wxuyw4&t=2632)]:** Complex matrix multiplications ke liye CPUs slow the. Nvidia ke CUDA (GPUs) aane se training time drastically reduce hua. Aaj specifically TPUs (Google), FPGAs, aur mobile/edge devices ke liye NPUs custom-built milte hain.
    
- **3. Frameworks & Libraries [[52:58](http://www.youtube.com/watch?v=fHF22Wxuyw4&t=3178)]:** Scratch se DL code likhna bohot tough tha. Google ne TensorFlow (later merged with Keras for simplicity) aur Facebook ne PyTorch (highly preferred in research) launch kiya, jisne implementation easy kar diya.
    
- **4. Pre-trained Architectures (Transfer Learning) [[58:45](http://www.youtube.com/watch?v=fHF22Wxuyw4&t=3525)]:** Har problem ke liye naya neural network architecture scratch se banana expensive aur time-consuming hai. ResNet (image), Transformer (text), YOLO (object detection) jaise state-of-the-art models ready-to-use available hain jinko aap directly apni problem par apply kar sakte ho.
    
- **5. The Community [[01:04:36](http://www.youtube.com/watch?v=fHF22Wxuyw4&t=3876)]:** Passionate researchers, engineers, aur Kaggle jaisi open communities ka collaborative effort jisne is technology ke potential ko actualize kiya.