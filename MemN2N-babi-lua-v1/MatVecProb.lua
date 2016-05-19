-- MatVecProb.lua

local MatVecProb, parent = torch.class('nn.MatVecProb', 'nn.Module')

function MatVecProb:__init(	do_transpose )
	parent.__init(self)

	self.gradInput = {torch.Tensor(), torch.Tensor()}

	self.do_transpose = do_transpose
end

function MatVecProb:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      self.gradInput[1]:cuda()
      self.gradInput[2]:cuda()
   end

   return self
end

function MatVecProb:updateOutput( input )
	-- longer mul shorter
	assert(#input == 2, 'input must be a pair of minibatch matrices')
	local a,b = table.unpack(input)

	if self.do_transpose then
		self.output = torch.zeros(a:size(1),a:size(3)):type(a:type())
		for i=1,a:size(1) do
			self.output[{i,{}}] = a[{i,{},{}}]:t() * b[{i,{}}]
		end
	else
		self.output = torch.zeros(a:size(1),a:size(2)):type(a:type())
		for i=1,a:size(1) do
			self.output[{i,{}}] = a[{i,{},{}}] * b[{i,{}}]:view(-1,1)
		end
	end
	if b:size(2) == 20 then
	end
	return self.output
end

function MatVecProb:updateGradInput( input, gradOutput )
	assert(#input == 2, 'input must be a paire of tensors')
	local a,b = table.unpack(input)
	self.gradInput[1]:resizeAs(a)
	self.gradInput[2]:resizeAs(b)

	local lg = (#gradOutput[{1,{}}])[1]
	local lb = (#b[{1,{}}])[1]

	for i=1,a:size(1) do
		if self.do_transpose then
			self.gradInput[1][{i,{},{}}] = b[{i,{}}]:view(-1,1) * gradOutput[{i,{}}]:view(1,-1)
			self.gradInput[2][{i,{}}] = a[{i,{},{}}] * gradOutput[{i,{}}]
		else

			self.gradInput[1][{i,{},{}}] = gradOutput[{i,{}}]:view(-1,1) * b[{i,{}}]:view(1,-1)
			self.gradInput[2][{i,{}}] = a[{i,{},{}}]:t() * gradOutput[{i,{}}]
		end
	end
	return self.gradInput

end