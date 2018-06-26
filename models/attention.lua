function attention.getfeatures(dataset,trainInds,sampleSeqLen)

    local permAllPersons = torch.randperm(trainInds:size(1))
    local personA = permAllPersons[1]--torch.floor(torch.rand(1)[1] * 2) + 1
    local personB = permAllPersons[2]--torch.floor(torch.rand(1)[1] * 2) + 1

    calculate1 = nn.ParallelTable()
    calculate1:add(nn.MetrixMultiply(opt.embeddingSize))
    calculate1:add(nn.Transpose({1,2}))

    calculate2 = nn.ConcatTable()
    calculate2:add(nn.MM())

    calculate = nn.Sequential()
    calculate:add(calculate1)
    calculate:add(calculate2)
    calculate:add(nn.SelectTable(1))
    calculate:add(nn.Tanh())

    attention1 = nn.ConcatTable()
    attention1:add(calculate)
    attention1:add(nn.SelectTable(1))
    attention1:add(nn.SelectTable(2))

    probe_seq = nn.Sequential()
    probe_seq:add(nn.SelectTable(1))
    probe_seq:add(nn.Max(2))
    probe_seq:add(nn.SoftMax())
    probe_seq:add(nn.Unsqueeze(1))
    
    probe1 = nn.ConcatTable()
    probe1:add(probe_seq)
    probe1:add(nn.SelectTable(2))
    probe1:cuda()

    probe5 = nn.ConcatTable()
    probe5:add(nn.MM())
    probe5:cuda()

    gallery_seq = nn.Sequential()
    gallery_seq:add(nn.SelectTable(1))
    gallery_seq:add(nn.Max(1))
    gallery_seq:add(nn.SoftMax())
    gallery_seq:add(nn.Unsqueeze(1))

    gallery1 = nn.ConcatTable()
    gallery1:add(gallery_seq)
    gallery1:add(nn.SelectTable(3))
    gallery1:cuda()

    gallery5 = nn.ConcatTable()
    gallery5:add(nn.MM())
    gallery5:cuda()

    probe = nn.Sequential()
    probe:add(probe1)
    probe:add(probe5)
    probe:cuda()

    gallery = nn.Sequential()
    gallery:add(gallery1)
    gallery:add(gallery5)
    gallery:cuda()

    attention4 = nn.ConcatTable()
    attention4:add(probe)
    attention4:add(gallery)
    attention4:cuda()

     local attention = nn.Sequential()
    attention:add(mlp2)
    attention:add(attention1)
    attention:add(attention4)
  	return attention
