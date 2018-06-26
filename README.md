# Two stream CNN-RNN network with attentive pooling for Person re-ID 
## Summary:
We perform video re-ID by using two streams, each stream is siamese based. RGB and optical flows are separately treat as input to learn spatial and temporal information separately. We apply attentive pooling on our base model available at:                     https://github.com/niallmcl/Recurrent-Convolutional-Video-ReID. In end we fuse the output of two streams to fully utilized features maps.  
## Step#1 Environment setting
- i): For this code to run you must have Torch7 installed with the nn, nnx, cunn, rnn, image,        	optim and cutorch pacakges.
- ii): You must have an Nvidia GPU in order to use CUDA. See http://torch.ch/ for details.
- iii): MATLAB R2015a.

## Step#2 Running the code : <br />
1: To run this code first download data sets avaiable at the following links:  <br />
 	 - iLIDS-VID: http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html  <br />
	 - PRID-2011: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/ <br />
	 - MARS: http://www.liangzheng.com.cn/Project/project_mars.html  <br />
2: Run datapreprocessing/computeOpticalFlow.m to generate optical flows. Optical Flow data will be generated in the same dir of  your datasets. datapreprocessing/PRID2011-OF-HVP for example..   <br />
3: Run traintest/videoReid.lua and edit the file line 70-77 to adjust the models and data paths. <br />
4: By default fusion file is run through videoReid.lua, but if you want to apply fusion within each stream then call fusion function inside CNN-RNN model in testtrain/train.lua. <br />
5: Example command-line options that will allow you to run the code in standard configuration  <br />
th reIdTrain.lua -nEpochs 600 -dataset 1 -dropoutFrac 0.6 -sampleSeqLength 16 -	samplingEpochs 100 -seed 1 -mode 'cnn-rgb' <br />
- NOTE: To train other datasets, change options -dataset. Similarly to run for only optical flow or for only RGB change - mode to cnn-rgb, cnn-optical, rgb-optical.  <br />
6: In case you encounter memory issues on your GPU, consider decreasing the cudnnWorkspaceLimit (512MB is default) <br />
7: After running the code weights files will store in your weights/folder. <br />
8: Run th reIdEval.lua to finally fusion and testing. <br />
9: If you have any further questions, please don't hesitate to contact me at wansar.mscs16seecs at seecs.edu.pk <br />
10: A slightly cleaned up implementation of our video re-id system is provided here. If pssible I will clean-up and improve the code in future....
