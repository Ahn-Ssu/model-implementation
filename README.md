# Deep Learning Model Implementations by PyTorch
* Convolution-based networks
    * Encoder architecture
    * Encoder-Decoder architecture
* Graph Convolution-based networks
* Recurrent Neural Networks



# Convolution-based
## Encoder
**CAMs) Learning Deep Features for Discriminative Localization**   
*Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba*   
[[paper](https://arxiv.org/abs/1512.04150)]    
CVPR 2016  

**DenseNet: Densely Connected Convolutional Networks**   
*Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger*  
[[paper](https://arxiv.org/abs/1608.06993)]    
CVPR 2017   

**MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**    
*Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam*   
[[paper](https://arxiv.org/abs/1704.04861)]   
CVPR 2017   

**MobileNetV2: Inverted Residuals and Linear Bottlenecks**    
*Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen*   
[[paper](https://arxiv.org/abs/1709.01507)]   
CVPR 2018   

**Squeeze-and-Excitation Networks**    
*Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu*   
[[paper](https://arxiv.org/abs/1801.04381)]   
CVPR 2018   

**EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**    
*Mingxing Tan, Quoc V. Le*   
[[paper](https://arxiv.org/abs/1905.11946)]   
ICML 2019   

**EfficientNetV2: Smaller Models and Faster Training**    
*Mingxing Tan, Quoc V. Le*   
[[paper](https://arxiv.org/abs/2104.00298)]   
ICML 2021   


## Encoder-Decoder
**FCN) Fully Convolutional Networks for Semantic Segmentation**   
*Jonathan Long, Evan Shelhamer, Trevor Darrell*   
[[paper](https://arxiv.org/abs/1411.4038)]    
CVPR 2015  

**U-Net: Convolutional Networks for Biomedical Image Segmentation**   
*Olaf Ronneberger, Philipp Fischer, Thomas Brox*   
[[paper](https://arxiv.org/abs/1505.04597)]    
MICCAI 2015 

**V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**    
*Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi*   
[[paper](https://arxiv.org/abs/1606.04797)]   
3DV 2016   

**Attention U-Net: Learning Where to Look for the Pancreas**    
*Ozan Oktay, et al.*   
[[paper](https://arxiv.org/abs/1804.03999)]   
MIDL 2018   

**SegResNet_VAE: 3D MRI brain tumor segmentation using autoencoder regularization**    
*Andriy Myronenko*   
[[paper](https://arxiv.org/abs/1810.11654]   
MICCAI 2018   


**UNet++: A Nested U-Net Architecture for Medical Image Segmentation**    
*Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang*   
[[paper](https://arxiv.org/abs/1807.10165)]   
DLMIA 2018   

**UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation**    
*Huimin Huang, Lanfen Lin, Ruofeng Tong, Hongjie Hu, Qiaowei Zhang, Yutaro Iwamoto, Xianhua Han, Yen-Wei Chen, Jian Wu*   
[[paper](https://arxiv.org/abs/2004.08790)]   
ICASSP(IEEE) 2020   

**DeepSEED: 3D Squeeze-and-Excitation Encoder-Decoder Convolutional Neural Networks for Pulmonary Nodule Detection**    
*Yuemeng Li, Yong Fan*   
[[paper](https://arxiv.org/abs/1904.03501)]   
ISBI 2020   

**DynUNet: Optimized U-Net for Brain Tumor Segmentation**    
*Michał Futrega, Alexandre Milesi, Michal Marcinkiewicz, Pablo Ribalta*   
[[paper](https://arxiv.org/abs/2110.03352)]   
MICCAI-BraTS 2021   


# Graph Convolution-based
**GCN) Semi-Supervised Classification with Graph Convolutional Networks**   
*Thomas N. Kipf, Max Welling*  
[[paper](https://arxiv.org/abs/1609.02907)]   
ICLR 2017   

 
**eGCN) Costless Performance Improvement in Machine Learning for Graph-Based Molecular Analysis**   
*Gyoun S. Na, Hyun Woo Kim, Hyunju Chang*  
[[paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00816)]   
Journal of Chemical Information and Modeling 2020   


**GAT_KAIST) Deeply learning molecular structure-property relationships using attention-and gate-augmented graph convolutional network**   
*Seongok Ryu, Jaechang Lim, Woo Youn Kim*  
[[paper](https://arxiv.org/abs/1805.10988)]   
CoRR 2018   


**GraphSAGE) Inductive Representation Learning on Large Graphs**   
*Will Hamilton, Zhitao Ying, Jure Leskovec*  
[[paper](https://arxiv.org/abs/1706.02216)]   
NIPS 2017   


**GIN) How Powerful are Graph Neural Networks?**   
*Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka*  
[[paper](https://arxiv.org/abs/1810.00826)]   
ICLR 2019   

# Recurrent Neural Networks  
**LSTM) Long short-term memory**   
*S Hochreiter, J Schmidhuber*  
[[paper](https://pubmed.ncbi.nlm.nih.gov/9377276/)], [[Post](https://pubmed.ncbi.nlm.nih.gov/9377276/)]    
Neural computation 1997   


**Seq2Seq w/ LSTM) Sequence to Sequence Learning with Neural Networks**   
*Ilya Sutskever, Oriol Vinyals, Quoc V. Le*  
[[paper](https://arxiv.org/abs/1409.3215)]    
NIPS 2014   

**Seq2Seq w/ Attention) Neural Machine Translation by Jointly Learning to Align and Translate**   
*Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio*  
[[paper](https://arxiv.org/abs/1409.0473)]    
NIPS 2014   

**Transformer) Attention Is All You Need**   
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*  
[[paper](https://arxiv.org/abs/1706.03762)]    
NIPS 2017   

## others
### Skip Connection
**Residual Connection) Deep Residual Learning for Image Recognition**   
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*  
[[paper](https://arxiv.org/abs/1512.03385)]    
CVPR 2016   

**Dense Connection) Densely Connected Convolutional Networks**   
*Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger*  
[[paper](https://arxiv.org/abs/1608.06993)]    
CVPR 2017   

### Loss
**Focal Loss for Dense Object Detection**   
*Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár*  
[[paper](https://arxiv.org/abs/1708.02002)]    
ICCV 2017   
