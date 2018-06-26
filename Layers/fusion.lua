function fusion(varargin) 

if ~isempty(gcp('nocreate')),
    delete(gcp)
end

opt = cnn_setup_environment();
opt.train.gpus = [1];
opt.cudnnWorkspaceLimit = [];
%add data name according to dataset

opt.dataSet = 'iLIDS-rgb'; 
opt.dataset1 = 'iLIDS-opt'

  

opt.dataDir = fullfile(opt.dataPath, opt.dataSet) ;
opt.splitDir = [opt.dataSet '_splits']; 

opt.inputdim  = [ 128, 224, 10] ;
opt.initMethod = '2sumAB';
opt.dropOutRatio = 0.9;

opt.train.fuseInto = 'spatial'; opt.train.fuseFrom = 'temporal';
opt.train.removeFuseFrom = 0 ;
opt.backpropFuseFrom = 1 ;
opt.nSplit = 1 ;
doSum = 0 ;
opt.train.learningRate =  1*[ 1e-3*ones(1,2) 1e-4*ones(1,1)  1e-5*ones(1,1) 1e-6*ones(1,1)]  ;
opt.train.cheapResize = 0 ;

correct=0
    ii=0
    for name in sorted(rgb.keys()):   
        r = rgb[name]
        o = opf[name]

        label = int(test_video[name])-1
                    
        video_level_preds[ii,:] = (r+o)
        video_level_labels[ii] = label
        ii+=1         
        if np.argmax(r+o) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
        

nFrames = 5;
if ~isempty(opt.train.gpus)
  opt.train.memoryMapFile = fullfile(tempdir, 'ramdisk', ['matconvnet' num2str(opt.train.gpus(1)) '.bin']) ;
end

opt.train.fusionType = 'conv';
opt.train.fusionLayer = {'relu5_3', 'relu5_3'; };

opt.expDir = fullfile(opt.dataDir, [opt.dataSet '-' model]) ;
opt.modelA = fullfile(opt.modelPath, [opt.dataSet 'iLIDS-rgb' num2str(opt.nSplit) '-dr0.9.mat']) ;
opt.modelB = fullfile(opt.modelPath, [opt.dataSet1 'iLIDS-opt' num2str(opt.nSplit) '-dr0.9.mat']) ;

opt.train.startEpoch = 1;
opt.train.epochStep = 1;
opt.train.epochFactor = 10;
opt.train.numEpochs = 1000 ;

[opt, varargin] = vl_argparse(opt, varargin) ;

opt.imdbPath = fullfile(opt.dataDir, [opt.dataSet '_split' num2str(opt.nSplit) 'imdb.mat']);


opt.train.batchSize = 100 ;
opt.train.numSubBatches = 50 / max(numel(opt.train.gpus),1); % lower this number if you have more GPU memory available

opt.train.saveAllPredScores = 1;
opt.train.denseEval = 1;

opt.train.plotDiagnostics = 0 ;
opt.train.continue = 1 ;
opt.train.prefetch = 1 ;
opt.train.expDir = opt.expDir ;

opt.train.numAugments = 1;
opt.train.frameSample = 'random';
opt.train.nFramesPerVid = 1;
opt.train.augmentation = 'ctr';

opt = vl_argparse(opt, varargin) ;


netA = load(opt.modelA) ;
netB = load(opt.modelB) ;
if isfield(netA, 'net'), netA=netA.net;end
if isfield(netB, 'net'), netB=netB.net;end


f = find(strcmp({netA.layers(:).type}, 'cnn.Loss'));
netA.layers(f(1)-1).name = 'prediction';
f = find(strcmp({netB.layers(:).type}, 'cnn.Loss'));
netB.layers(f(1)-1).name = 'prediction';

fusionLayerA = []; fusionLayerB = [];
if ~isempty(opt.train.fusionLayer)
for i=1:numel(netA.layers)
 if isfield(netA.layers(i),'name') && any(strcmp(netA.layers(i).name,opt.train.fusionLayer(:,1)))
   fusionLayerA = [fusionLayerA i]; 
 end                
end
for i=1:numel(netB.layers)
 if  isfield(netB.layers(i),'name') && any(strcmp(netB.layers(i).name,opt.train.fusionLayer(:,2)))
   fusionLayerB = [fusionLayerB i]; 
 end                
end
end

netA.meta.normalization.averageImage = mean(mean(netA.meta.normalization.averageImage, 1), 2);
netB.meta.normalization.averageImage = mean(mean(netB.meta.normalization.averageImage, 1), 2);
netB.meta.normalization.averageImage = gather(cat(3,netB.meta.normalization.averageImage, netA.meta.normalization.averageImage));

% rename layers, params and vars
for x=1:numel(netA.layers)
  if isfield(netA.layers(x), 'name'), netA.layers(x).name = [netA.layers(x).name '_spatial'] ;  end
end
for x=1:numel(netB.layers)
  if isfield(netB.layers(x), 'name'), netB.layers(x).name = [netB.layers(x).name '_temporal']; end
end

% inject conv fusion layer
if addConv3D & any(~cellfun(@isempty,(strfind(opt.train.fusionLayer, 'prediction'))))
  if strcmp(opt.train.fuseInto,'temporal')
    [ netB ] = insert_conv_layers( netB, fusionLayerB(end), 'initMethod', opt.initMethod );
  else
    [ netA ] = insert_conv_layers( netA, fusionLayerA(end), 'initMethod', opt.initMethod );
  end
end
if ~addConv3D && ~doSum 
  if strcmp(opt.train.fuseInto,'temporal')
    [ netB ] = insert_conv_layers( netB, fusionLayerB, 'initMethod', opt.initMethod );
  else
    [ netA ] = insert_conv_layers( netA, fusionLayerA, 'initMethod', opt.initMethod );
  end
end

if opt.train.removeFuseFrom, 
  switch opt.train.fuseFrom
    case 'spatial'
      netA.layers = netA.layers(1:fusionLayerA(end)); netA.rebuild;
    case'temporal'
      netB.layers = netB.layers(1:fusionLayerB(end)); netB.rebuild;
  end
end


  if doSum
    block = cnn.Sum() ;
    net.addLayerAt(i_fusion(end), name_concat, block, ...
               [net.layers(strcmp({net.layers.name},[opt.train.fusionLayer{i,1} '_spatial'])).outputs ...
                net.layers(strcmp({net.layers.name},[opt.train.fusionLayer{i,2} '_temporal'])).outputs], ...
                name_concat) ;   
              

  else
    block = cnn.Concat() ;
    net.addLayerAt(i_fusion(end), name_concat, block, ...
               [net.layers(strcmp({net.layers.name},[opt.train.fusionLayer{i,1} '_spatial'])).outputs ...
                net.layers(strcmp({net.layers.name},[opt.train.fusionLayer{i,2} '_temporal'])).outputs], ...
                name_concat) ;   
  end

  
end




for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'cnn.DropOut')
    net.layers(l).block.rate = opt.dropOutRatio;
  end
end

net.layers(~cellfun('isempty', strfind({net.layers(:).name}, 'err'))) = [] ;
net.rebuild() ;

opt.train.derOutputs = {} ;
for l=1:numel(net.layers)
  if isa(net.layers(l).block, 'cnn.Loss') && isempty(strfind(net.layers(l).block.loss, 'err'))
    if opt.backpropFuseFrom || ~isempty(strfind(net.layers(l).name, opt.train.fuseInto ))
           opt.train.derOutputs = [opt.train.derOutputs, net.layers(l).outputs, {1}] ;
    end
     net.addLayer(['err1_' net.layers(l).name(end-7:end) ], cnn.Loss('loss', 'class error'), ...
             net.layers(l).inputs, 'error') ;
  end
end
 
feature-info = cnn_train_dag(net, imdb, fn, opt.train) ;
return feature-info

