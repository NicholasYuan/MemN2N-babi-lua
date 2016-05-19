-- CrossEntropyLossNB
local CrossEntropyLossNB, parent = torch.class("nn.CrossEntropyLossNB", "nn.Module")

function CrossEntropyLossNB:__init( size_average, do_softmax_brop )
	if do_softmax_brop ~= nil then
		self.do_softmax_brop = do_softmax_brop
	else
		self.do_softmax_brop = false
	end
	
	self.eps = 0.0000001
	self.size_average = size_average
end

function CrossEntropyLossNB:updateOutput( input_target )
	input, target = unpack(input_target)

	target = target:long()
	local z = target:view(1,-1)
	
	x = input:t():gather(1,z)
	cost = torch.sum(-torch.log(x))
	if self.size_average then
		cost = cost / input:size(1)
	end
	return cost
end

function CrossEntropyLossNB:updateGradInput( input, target )
	self.gradInput = torch.Tensor(input:t():size())
	target = target:long()
	local z = target:view(1,-1)
	local _input = input:t()

	local x = _input:gather(1,z)

	if self.do_softmax_brop then
		self.gradInput:copy(_input)
		local xtmp = x-1
		self.gradInput:scatter(1,z,xtmp)
	else
		self.gradInput = torch.zeros(_input:size())
		local xtmp = -torch.cdiv(torch.ones(x:size()), x + self.eps)
		self.gradInput = self.gradInput:scatter(1,z,xtmp)
	end
	if self.size_average then
		self.gradInput = self.gradInput / _input:size(2)
	end
	self.gradInput = self.gradInput:t()
	return self.gradInput
end