
--
-- Copyright (c) 2016 Niall McLaughlin, CSIT, Queen's University Belfast, UK
-- Contact: nmclaughlin02@qub.ac.uk
-- If you use this code please cite:
-- "Recurrent Convolutional Network for Video-based Person Re-Identification",
-- N McLaughlin, J Martinez Del Rincon, P Miller, 
-- IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
-- 
-- This software is licensed for research and non-commercial use only.
-- 
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
-- THE SOFTWARE.

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'cunn'
require 'cutorch'
require 'image'
require 'paths'
require 'rnn'
require 'inn'

require 'layers/MetrixMultiply'
require 'train'
require 'test'
require 'fusion'

local datasetUtils = require 'datasets/datasetUtils'
local prepDataset = require 'datasets/prepareDataset'

-- set the GPU
cutorch.setDevice(1)

cmd = torch.CmdLine()
cmd:option('-nEpochs',600,'number of training epochs')
cmd:option('-dataset',1,'1 -  ilids, 2 - prid, 3 - mars')
cmd:option('-dirRGB','','dir path to the sequences of original datasets')
cmd:option('-dirOF','','dir path to the sequences of optical flow')
cmd:option('-sampleSeqLength',16,'length of sequence to train network')
cmd:option('-gradClip',5,'magnitude of clip on the RNN gradient')
cmd:option('-saveFileName','basicnet','name to save dataset file')
cmd:option('-usePredefinedSplit',false,'Use predefined test/training split loaded from a file')
cmd:option('-dropoutFrac',0.6,'fraction of dropout to use between layers')
cmd:option('-dropoutFracRNN',0.6,'fraction of dropout to use between RNN layers')
cmd:option('-samplingEpochs',100,'how often to compute the CMC curve - dont compute too much - its slow!')
cmd:option('-disableOpticalFlow',false,'use optical flow features or not')
cmd:option('-seed',1,'random seed')
cmd:option('-learningRate',1e-3)
cmd:option('-momentum',0.9)
cmd:option('-nConvFilters',32)
cmd:option('-embeddingSize',128)
cmd:option('-hingeMargin',3)
cmd:option('-mode','rgb-optical','three mode: cnn-rgb, cnn-optical, rgb-optical')

opt = cmd:parse(arg)
print(opt)

opt.spatial = 0
opt.temporal = 0

if opt.mode == 'cnn-rgb' then
    require 'models/cnn-rnn'
    opt.spatial = 1
    opt.temporal = 0
elseif opt.mode == 'cnn-optical' then
    require 'models/cnn-rnn'
    opt.spatial = 0
    opt.temporal = 1
elseif opt.mode == 'rgb-optical' then
    require 'models/spatial_temporal'
    opt.spatial = 1
    opt.temporal = 1
else
    print('Unknown mode')
    os.exit(0)
end

function isnan(z)
    return z ~= z
end

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- change these paths to point to the place where you store i-lids or prid datasets
homeDir = paths.home
filePrefix='.png'

if opt.dataset == 1 then
    seqRootRGB = 'data/iLIDS-VIDNEW/i-LIDS-VID/sequences'
    seqRootOF = 'data/iLIDS-VIDNEW/i-LIDS-VID-OF-HVP/sequences/'
elseif opt.dataset == 2 then
    seqRootRGB = homeDir .. 'data/PRID2011/multi_shot/'
    seqRootOF = homeDir .. 'data/PRID2011-OF-HVP/multi_shot/'
elseif opt.dataset == 3 then
    seqRootRGB = homeDir .. 'data/myMars/sequences/'
    seqRootOF = homeDir .. 'data/myMars/MARS-OF-HVP/sequences/'
    filePrefix='.jpg'
else
    print('Unknown datasets')
    os.exit(0)
end

if opt.seqRootRGB and opt.seqRootOF then
    seqRootRGB = opt.seqRootRGB
    seqRootOF = opt.seqRootOF
end


print('loading Dataset - ',seqRootRGB,seqRootOF)
dataset = prepDataset.prepareDataset(seqRootRGB,seqRootOF,filePrefix)
print('dataset loaded',#dataset,seqRootRGB,seqRootOF)

if opt.usePredefinedSplit then
    -- useful for debugging to run with exactly the same test/train split
    print('loading predefined test/training split')
    local datasetSplit
    if opt.dataset == 1 then
        datasetSplit = torch.load('./data/iLIDS-VIDNEW/dataSplit.th7')
    if opt.dataset ==2 then
        datasetSplit = torch.load('./data/PRID2011/dataSplit_PRID2011.th7')
	if opt.dataset ==3 then
        datasetSplit = torch.load('./data/myMars/dataSplit_MARS.th7')
    end    
    testInds = datasetSplit.testInds
    trainInds = datasetSplit.trainInds
else
    print('randomizing test/training split')
    trainInds,testInds = datasetUtils.partitionDataset(#dataset,0.5)
end

-- build the model
fullModel,criterion,Combined_CNN_RNN,baseCNN = buildModel_MeanPool_RNN(opt,16,opt.nConvFilters,opt.nConvFilters,trainInds:size(1))

-- train the model
trainedModel,trainedConvnet,trainedBaseNet = trainSequence(fullModel,Combined_CNN_RNN,baseCNN,criterion,dataset,nSamplesPerPerson,trainInds,testInds,nEpochs)
--fusion
 Foutput=  fusion(gradOutput) 

-- save the Model and Convnet (which is part of the model) to a file
saveFileNameModel = '/home/data/TSCRAN/trainedNets/fullModel_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameModel,trainedModel)
saveFileNameConvnet = '/home/data/TSCRAN/trainedNets/convNet_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameConvnet,trainedConvnet)
saveFileNameBasenet = '/home/data/TSCRAN/trainedNets/baseNet_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameBasenet,trainedBaseNet)

------------------------------------------------------------------------------------------------------------------------------------
-- Evaluation
------------------------------------------------------------------------------------------------------------------------------------

trainedConvnet:evaluate()
nTestImages = {1,2,4,8,16,32,64,128}

for n = 1,#nTestImages do
    print('test multiple images '..nTestImages[n])
    -- default method of computing CMC curve
    computeCMC_MeanPool_RNN(dataset,testInds,trainedConvnet,128,nTestImages[n])
end
