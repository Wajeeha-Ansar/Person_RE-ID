local MetrixMultiply, Parent = torch.class('nn.MetrixMultiply', 'nn.Module')

function MetrixMultiply:__init(dim)
    Parent.__init(self)
    self.dim = dim
    self.weight = torch.Tensor(dim,dim)
    self.gradWeight = torch.Tensor(dim,dim)
    self:reset()
end

function MetrixMultiply:reset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1/math.sqrt(self.dim*self.dim)
    end
    if nn.oldSeed then
        self.weight:apply(function()
            return torch.uniform(-stdv, stdv)
        end)
    else
        self.weight:uniform(-stdv, stdv)
    end
end

function MetrixMultiply:updateOutput(input)
    assert(input:nDimension() == 2 or input:nDimension() == 3, 'input tensors must be 2D or 3D')

    if input:nDimension() == 2 then
        assert(input:size(2) == self.weight:size(1), 'matrix sizes do not match')

        self.output:resize(input:size(1), self.weight:size(2))
        self.output:mm(input, self.weight)
    else
        assert(input:nDimension() == 3, 'second input tensor must be 3D')

        assert(input:size(3) == self.weight:size(1), 'matrix sizes do not match')

        self.output:resize(input:size(1), input:size(2), self.weight:size(3))
        self.output:mm(input, self.weight)
    end

    return self.output
end

function MetrixMultiply:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput or input.new()

    self.gradInput:resizeAs(input)

    assert(gradOutput:nDimension() == 2 or gradOutput:nDimension() == 3, 'arguments must be a 2D or 3D Tensor')

    self.gradInput = torch.mm(gradOutput, self.weight)

    return self.gradInput
end

function MetrixMultiply:accGradParameters(input, gradOutput)
    self.gradWeight:resize(self.dim,self.dim)
    local a = input

    assert(gradOutput:nDimension() == 2 or gradOutput:nDimension() == 3, 'arguments must be a 2D or 3D Tensor')

    local h_dim, w_dim, f
    if gradOutput:nDimension() == 2 then
        assert(input:nDimension() == 2, 'first input tensor must be 2D')

        h_dim, w_dim = 1, 2
        f = "mm"
    else
        assert(a:nDimension() == 3, 'first input tensor must be 3D')
        assert(b:nDimension() == 3, 'second input tensor must be 3D')

        h_dim, w_dim = 2, 3
        f = "bmm"
    end

    a = a:transpose(h_dim, w_dim)
    self.gradWeight[f](self.gradWeight, a, gradOutput)
end
