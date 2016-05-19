-- EleMult
local EleMult, parent = torch.class('nn.EleMult', 'nn.Module')

function EleMult:__init(w)
   parent.__init(self)
   self.ww = w:clone()
   self.count = 0
end

function EleMult:makeInputContiguous( input )
	
	if (not input:isContiguous()) then
		self.copiedInput = true
		self._input:resize(input:size()):copy(input)
		return self._input
	end
	self.copiedInput = false
	return input
end

function EleMult:updateOutput( input )
	self.output = torch.Tensor(input:size())
	self.output:zero()
	input = self:makeInputContiguous(input)

	if input:dim() == 3 then
		for i=1,input:size(1) do
			self.output[{i,{},{}}] = torch.cmul(self.ww,input[{i,{},{}}])
		end
	elseif input:dim() == 4 then
		for i=1,input:size(1) do
			for j=1,input:size(2) do
				self.output[{i,j,{},{}}] = torch.cmul(self.ww, input[{i,j,{},{}}])
			end
		end
	end
	return self.output
end

function EleMult:updateGradInput( input, gradOutput )
	self.gradInput = torch.Tensor(input:size()):type(input:type())
	self.gradInput:zero()
	input = self.copiedInput and self._input or input

	if input:dim() == 3 then
		for i=1,input:size(1) do
			self.gradInput[{i,{},{}}] = torch.cmul(self.ww,gradOutput[{i,{},{}}])
		end
	elseif input:dim() == 4 then
		for i=1,input:size(1) do
			for j=1,input:size(2) do
				self.gradInput[{i,j,{},{}}] = torch.cmul(self.ww,gradOutput[{i,j,{},{}}])
			end
		end

	end
	return self.gradInput
end