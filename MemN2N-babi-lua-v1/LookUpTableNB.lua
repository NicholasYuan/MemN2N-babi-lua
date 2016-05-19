-- LookUpTableNB
local LookUpTableNB, parent = torch.class('nn.LookUpTableNB', 'nn.Module')

function LookUpTableNB:__init(nIndex, nOutput, max_grad_norm)
   parent.__init(self)
   self.input_size = nIndex
   self.output_dim = nOutput
   self.weight = torch.randn(nIndex, nOutput) * 0.1
   self.gradWeight = torch.zeros(nIndex, nOutput)

   self.max_grad_norm = max_grad_norm
   self:reset()
end

function LookUpTableNB:backCompatibility()
   self._count = self._count or torch.IntTensor()
   self._input = self._input or torch.LongTensor()

   if not self.shouldScaleGradByFreq then
      self.shouldScaleGradByFreq = false
   end
end

function LookUpTableNB:makeInputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end

function LookUpTableNB:updateOutput(input)
   self:backCompatibility()
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
      self.output = self.output:view(input:size(1), self.output_dim)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.output_dim)
      -- print(self.output[{{},{},1}])
   elseif input:dim() == 3 then
  	   self.output:index(self.weight, 1, input:view(-1))
  	   self.output = self.output:view(input:size(1), input:size(2), input:size(3), self.output_dim)
      else
      error("input must be a vector or matrix or 3D matrix")
   end

   return self.output

end

function LookUpTableNB:accGradParameters(input, gradOutput, scale)
	self:backCompatibility()
	input = self.copiedInput and self._input or input
	if input:dim() == 2 or input:dim() == 3 then
   	-- input = input:view(-1)
	elseif input:dim() ~= 1 then
   	error("input must be a vector or matrix or 3D matrix")
	end

	if not gradOutput:isContiguous() then
    	self._gradOutput = self._gradOutput or gradOutput.new()
    	self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
    	gradOutput = self._gradOutput
	end
	scale = scale or 1
	if input:dim() == 1 then
		for i=1,input:size(1) do
			self.gradWeight[{input[i],{}}] = self.gradWeight[{input[i],{}}] + gradOutput[{i,{}}] * scale
		end
	elseif input:dim() == 2 then
	   	for i=1,input:size(1) do
	   		for j=1,input:size(2) do
	   			self.gradWeight[{input[{i,j}],{}}] = self.gradWeight[{input[{i,j}],{}}] + gradOutput[{i,j,{}}] * scale
	   		end
	   	end
	elseif input:dim() == 3 then
		for i=1,input:size(1) do
			for j=1,input:size(2) do
				for k=1,input:size(3) do
					self.gradWeight[{input[{i,j,k}],{}}] = self.gradWeight[{input[{i,j,k}],{}}] + gradOutput[{i,j,k,{}}] *scale
				end
			end
		end
	end
	local norm = self.gradWeight:norm()
	if norm > self.max_grad_norm then
		self.gradWeight = self.gradWeight:mul(self.max_grad_norm/norm)
	end
end

function LookUpTableNB:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.weight.new()
      self._indices = self.weight.new()
      self._count = self.weight.new()
      self._input = self.weight.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end

   return self
end
